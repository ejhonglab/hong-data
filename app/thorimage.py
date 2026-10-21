"""Light-touch parsing of a ThorImage ``Experiment.xml``.

We only pull a handful of attributes that are useful to show next to a dataset.
Everything is optional: any missing element is simply skipped, and the file is stored
verbatim regardless. The laser wavelength (``MultiPhotonLaser``) is deliberately not
read: it does not follow changes made to the wavelength at the laser.
"""
from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional


class ExperimentXmlError(ValueError):
    pass


_YAML_NAME = re.compile(r"[^\s\"'<>|*?]+\.ya?ml\b", re.IGNORECASE)


def _attr(el: Optional[ET.Element], name: str) -> Optional[str]:
    if el is None:
        return None
    v = el.get(name)
    return v if v not in (None, "") else None


def _num(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_experiment_xml(text: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ExperimentXmlError(f"not well-formed XML: {e}")
    if root.tag != "ThorImageExperiment":
        raise ExperimentXmlError(
            f"root element is <{root.tag}>, expected <ThorImageExperiment>; is this a ThorImage Experiment.xml?"
        )
    info: dict[str, Any] = {}
    name = root.find("Name")
    info["name"] = _attr(name, "name")
    info["path"] = _attr(name, "path")
    date = root.find("Date")
    info["date"] = _attr(date, "date")
    sw = root.find("Software")
    info["software_version"] = _attr(sw, "version")
    lsm = root.find("LSM")
    if lsm is not None:
        info["lsm"] = {
            k: _num(_attr(lsm, k))
            for k in ("pixelX", "pixelY", "frameRate", "pixelSizeUM", "averageNum", "averageMode", "widthUM", "heightUM")
            if _attr(lsm, k) is not None
        }
    z = root.find("ZStage")
    if z is not None:
        info["zstage"] = {k: _num(_attr(z, k)) for k in ("steps", "stepSizeUM") if _attr(z, k) is not None}
    stream = root.find("Streaming")
    if stream is not None:
        info["streaming"] = {k: _num(_attr(stream, k)) for k in ("frames", "enable") if _attr(stream, k) is not None}
    mag = root.find("Magnification")
    info["magnification"] = _num(_attr(mag, "mag"))
    pmt = root.find("PMT")
    if pmt is not None:
        info["pmt"] = {k: _num(v) for k, v in pmt.attrib.items() if k.startswith(("gain", "enable"))}
    pockels = root.find("Pockels")
    if pockels is not None:
        info["pockels"] = {k: _num(_attr(pockels, k)) for k in ("start", "stop") if _attr(pockels, k) is not None}
    comments = root.find("Comments")
    info["comments"] = _attr(comments, "text")
    wl = root.find("Wavelengths")
    if wl is not None:
        info["channels"] = [w.get("name") for w in wl.findall("Wavelength") if w.get("name")]
    info = {k: v for k, v in info.items() if v not in (None, {}, [])}
    return {"info": info}


def raw_layout(text: str) -> dict[str, Any]:
    """How the .raw file of a ThorImage recording is laid out, as hong2p.thor reads it.

    Planes of ``y`` × ``x`` little-endian uint16 are ordered by time, then Z (``z`` planes
    then ``flyback`` frames in each volume, with fast Z), then channel (``c``: PMT channels
    A and B enabled, gain or not).

    Returns:
        dict with x, y, z, flyback, c, and fps: planes per second after averaging (flyback
        frames counted), or None if not given.

    Raises:
        ExperimentXmlError: if the XML is not readable or has no frame size.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ExperimentXmlError(f"not well-formed XML: {e}")
    lsm = root.find("LSM")
    x, y = _num(_attr(lsm, "pixelX")), _num(_attr(lsm, "pixelY"))
    if not x or not y:
        raise ExperimentXmlError("no frame size (LSM pixelX and pixelY) in the Experiment.xml")
    zstages = root.findall("ZStage")
    zstage = next((z for z in zstages if z.get("name")), zstages[0] if zstages else None)
    z = int(_num(_attr(zstage, "steps")) or 1)
    stream = root.find("Streaming")
    fast_z = _attr(stream, "enable") == "1" and _attr(stream, "zFastEnable") == "1"
    flyback = int(_num(_attr(stream, "flybackFrames")) or 0) if fast_z else 0
    pmt = root.find("PMT")
    c = sum(_attr(pmt, f"enable{ch}") == "1" for ch in "AB") or 1
    fps = _num(_attr(lsm, "frameRate"))
    if fps and _attr(lsm, "averageMode") not in (None, "0"):
        fps /= _num(_attr(lsm, "averageNum")) or 1
    return {"x": int(x), "y": int(y), "z": z, "flyback": flyback, "c": c, "fps": fps}


def recorded_at(text: str) -> Optional[dt.datetime]:
    """When a recording was made, by its Experiment.xml (``<Date date="11/19/2023 14:23:10">``,
    the acquisition computer's time); None if it does not say, or cannot be read."""
    try:
        date = _attr(ET.fromstring(text).find("Date"), "date")
    except ET.ParseError:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime((date or "").strip(), fmt)
        except ValueError:
            continue
    return None


def notes_yaml_names(text: str) -> list[str]:
    """File names of the YAMLs mentioned in an Experiment.xml's notes (where hong2p looks for the stimulus file).

    Returns:
        names without their folders (which may be Windows paths), or [] if the XML
        cannot be read.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    notes = " ".join(t for t in (_attr(root.find("ExperimentNotes"), "text"), _attr(root.find("Comments"), "text")) if t)
    return list(dict.fromkeys(m.replace("\\", "/").rsplit("/", 1)[-1] for m in _YAML_NAME.findall(notes)))
