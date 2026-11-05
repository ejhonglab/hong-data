"""Light-touch parsing of a ThorImage ``Experiment.xml``.

We only pull a handful of attributes that are useful to show next to a dataset.
Everything is optional: any missing element is simply skipped, and the file is stored
verbatim regardless. The laser wavelength (``MultiPhotonLaser``) is deliberately not
read: it does not follow changes made to the wavelength at the laser.
"""
from __future__ import annotations

import datetime as dt
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Optional

from hong2p import thor


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
    """How the .raw file of a ThorImage recording is laid out, read with hong2p.thor.

    Planes of ``y`` × ``x`` little-endian uint16 are ordered by time, then Z (``z`` planes
    then ``flyback`` frames in each volume, with fast Z), then channel (``c``: PMT channels
    A and B enabled, gain or not).

    Returns:
        dict with x, y, z, flyback, c, and fps: planes per second after averaging (flyback
        frames counted; averaging configured with fast Z is not done by ThorImage), or
        None if not given.

    Raises:
        ExperimentXmlError: if the XML is not readable or does not give the layout.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ExperimentXmlError(f"not well-formed XML: {e}")
    try:
        (x, y), z, c = thor.get_thorimage_dims(root)
        flyback = thor.get_thorimage_n_flyback_xml(root)
    except (AttributeError, KeyError, ValueError, AssertionError) as e:
        raise ExperimentXmlError(f"the Experiment.xml does not give the .raw file's layout (frame size, ZStage, "
                                 f"Streaming and PMT): {type(e).__name__} {e}")
    if not (x and y):
        raise ExperimentXmlError("no frame size (LSM pixelX and pixelY) in the Experiment.xml")
    if not c:
        raise ExperimentXmlError("no PMT channel (A or B) is enabled in the Experiment.xml")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # that averaging was configured with fast Z
            fps = thor.get_thorimage_fps_xml(root)
    except (AttributeError, KeyError, ValueError):
        fps = None
    return {"x": x, "y": y, "z": z, "flyback": flyback, "c": c, "fps": fps}


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
