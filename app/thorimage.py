"""Light-touch parsing of a ThorImage ``Experiment.xml``.

We only pull a handful of attributes that are useful to show next to a dataset (and
to suggest form values such as the laser wavelength). Everything is optional: any
missing element is simply skipped, and the file is stored verbatim regardless.
"""
from __future__ import annotations

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
    laser = root.find("MultiPhotonLaser")
    laser_pos = _num(_attr(laser, "position"))
    info["laser_position"] = laser_pos
    suggestions: dict[str, Any] = {}
    if laser_pos is not None and 600 <= laser_pos <= 1400:
        suggestions["wavelength_nm"] = laser_pos
    comments = root.find("Comments")
    info["comments"] = _attr(comments, "text")
    wl = root.find("Wavelengths")
    if wl is not None:
        info["channels"] = [w.get("name") for w in wl.findall("Wavelength") if w.get("name")]
    info = {k: v for k, v in info.items() if v not in (None, {}, [])}
    return {"info": info, "suggestions": suggestions}


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
