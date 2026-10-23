"""Experiment metadata entered in the upload form, and README.md generation."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_INDICATORS = ["GCaMP6f", "GCaMP6s", "GCaMP6m", "GCaMP7f", "GCaMP7s", "GCaMP8f", "GCaMP8m", "GCaMP8s"]
DEFAULT_CODING_UNITS = ["glomeruli", "KCs"]
DEFAULT_RECORDING_SYSTEMS = ["upstairs-left 2p", "upstairs-right 2p", "downstairs 2p"]
DEFAULT_WAVELENGTH_NM = 925.0
REQUIRED_FIELDS = ("uploader_name", "indicator", "driver", "coding_unit")


def _clean(v):
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


class Metadata(BaseModel):
    uploader_name: str = Field(min_length=1)
    indicator: str = Field(min_length=1)
    driver: str = Field(min_length=1)
    coding_unit: str = Field(min_length=1)

    genotype: Optional[str] = None
    age_min_days: Optional[float] = Field(default=None, ge=0)
    age_max_days: Optional[float] = Field(default=None, ge=0)
    power_mw: Optional[float] = Field(default=None, gt=0)
    power_tolerance_mw: Optional[float] = Field(default=None, ge=0)
    power_measured_same_day: Optional[bool] = None
    wavelength_nm: Optional[float] = Field(default=DEFAULT_WAVELENGTH_NM, gt=0)
    source_code_url: Optional[str] = None
    raw_data_path: Optional[str] = None
    mocorr_data_path: Optional[str] = None
    panel_names: list[str] = Field(default_factory=list)
    recording_system: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "uploader_name", "indicator", "driver", "coding_unit", "genotype", "source_code_url",
        "raw_data_path", "mocorr_data_path", "recording_system", "notes", mode="before",
    )
    @classmethod
    def _strip(cls, v):
        return _clean(v)

    @field_validator("panel_names", mode="before")
    @classmethod
    def _panels(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.split(",")
        out = []
        for x in v:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        return out

    @model_validator(mode="after")
    def _cross(self):
        if self.age_min_days is not None and self.age_max_days is not None and self.age_max_days < self.age_min_days:
            raise ValueError("fly age range: max must be >= min")
        if self.power_mw is None and (self.power_tolerance_mw is not None or self.power_measured_same_day is not None):
            raise ValueError("power tolerance / same-day measurement need a power value")
        return self


def _age_str(m: Metadata) -> Optional[str]:
    lo, hi = m.age_min_days, m.age_max_days
    if lo is None and hi is None:
        return None
    f = lambda x: f"{x:g}"
    if lo is not None and hi is not None:
        return f"{f(lo)} d" if lo == hi else f"{f(lo)}–{f(hi)} d"
    return f"≥ {f(lo)} d" if lo is not None else f"≤ {f(hi)} d"


def _power_str(m: Metadata) -> Optional[str]:
    if m.power_mw is None:
        return None
    s = f"{m.power_mw:g} mW at the fly"
    if m.power_tolerance_mw is not None:
        s += f" (±{m.power_tolerance_mw:g} mW)"
    if m.power_measured_same_day is True:
        s += ", measured on the day of the experiments"
    elif m.power_measured_same_day is False:
        s += ", measured on a different day"
    return s


def dataset_title(m: Metadata, summary: dict) -> str:
    panels = m.panel_names or summary.get("panel_names") or []
    panel_part = ", ".join(panels) if panels else "odor panel"
    return f"{panel_part}: {m.driver} > {m.indicator} ({m.coding_unit})"


def generate_readme(m: Metadata, summary: dict, files: dict[str, Any], *, created_at: Optional[dt.datetime] = None) -> str:
    """Concise README.md for a dataset.

    ``files`` keys: data (original file name), odor_yamls (list of names),
    experiment_xml (name or None), sha256 (hex of the data file), and optionally
    sources (file name -> the path on tensor it was copied from).
    """
    created_at = created_at or dt.datetime.now()
    flies = summary.get("flies", [])
    n_flies = summary.get("n_flies", len(flies))
    first, last = summary.get("first_date"), summary.get("last_date")
    if first and last:
        dates = first if first == last else f"{first} to {last}"
    else:
        dates = "unknown dates"
    rep_min, rep_max = summary.get("repeats_min"), summary.get("repeats_max")
    rep = f"{rep_min}" if rep_min == rep_max else f"{rep_min}–{rep_max}"
    roi_kind = summary.get("roi_kind", "")
    roi_word = {"glomeruli": "glomerulus ROIs", "integers": "numbered ROIs", "mixed": "ROIs"}.get(roi_kind, "ROIs")

    lines = [f"# {dataset_title(m, summary)}", ""]
    lines.append(
        f"Mean responses of {m.coding_unit} ({roi_word}) to {summary.get('n_odors', '?')} odors "
        f"in {n_flies} {'fly' if n_flies == 1 else 'flies'} ({dates}); "
        f"{summary.get('n_units', '?')} ROI columns, {rep} trial(s) per odor. "
        f"{m.driver} driving {m.indicator}."
    )
    lines.append("")
    rows = [
        ("Uploaded by", f"{m.uploader_name} ({created_at:%Y-%m-%d})"),
        ("Panel(s)", ", ".join(m.panel_names) if m.panel_names else None),
        ("Driver", m.driver),
        ("Indicator", m.indicator),
        ("Full genotype", m.genotype),
        ("Coding unit", m.coding_unit),
        ("Fly age", _age_str(m)),
        ("Excitation", (f"{m.wavelength_nm:g} nm" if m.wavelength_nm else None)),
        ("Power", _power_str(m)),
        ("Rig", m.recording_system),
        ("Raw data (tensor)", f"`{m.raw_data_path}`" if m.raw_data_path else None),
        ("Motion-corrected data (tensor)", f"`{m.mocorr_data_path}`" if m.mocorr_data_path else None),
        ("Analysis code", m.source_code_url),
    ]
    lines += ["| | |", "|---|---|"]
    for k, v in rows:
        if v:
            lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Data")
    lines.append(f"- `{files.get('data')}` ({summary.get('source_format', 'matrix')}; sha256 {str(files.get('sha256', ''))[:12]})")
    lines.append(
        f"- columns: {' / '.join(summary.get('col_levels', []))}; rows: {' / '.join(summary.get('row_levels', []))}"
    )
    fly_strs = [f"{f['date']} fly {f['fly_num']} ({f['n_rois']} ROIs)" for f in flies]
    if fly_strs:
        lines.append("- flies: " + "; ".join(fly_strs))
    nan = summary.get("nan_fraction", 0.0) or 0.0
    if nan > 0:
        lines.append(f"- missing cells: {nan:.1%}")
    lines.append("")

    lines.append("## Odors")
    for p in summary.get("panels", []):
        label = p["panel"] or "(no panel level)"
        lines.append(f"- **{label}** ({len(p['odors'])} odors, {p['n_trials']} trials): " + ", ".join(p["odors"]))
    lines.append("")

    sources = files.get("sources") or {}
    origin = lambda name: f" (from tensor: `{sources[name]}`)" if sources.get(name) else ""
    extra_files = []
    for y in files.get("odor_yamls", []) or []:
        extra_files.append(f"- odor metadata: `{y}`{origin(y)}")
    if files.get("experiment_xml"):
        extra_files.append(f"- ThorImage: `{files['experiment_xml']}`{origin(files['experiment_xml'])}")
    if extra_files:
        lines.append("## Files")
        lines += extra_files
        lines.append("")

    if m.notes:
        lines.append("## Notes")
        lines.append(m.notes.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
