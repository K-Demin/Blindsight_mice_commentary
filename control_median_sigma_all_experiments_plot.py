#!/usr/bin/env python3
"""Unequal-variance SDT analysis and manuscript figure.

This script generates the unequal-variance SDT figure and the corresponding
analysis table for the commentary by Demin, Miyoshi, and Lau.

1. Extract right-choice rates for Blank, RO, CO, and RC from the prepared CSVs.
2. Retain ordinary percentage-rounding cases (99-101%), but exclude non-extractable rows.
3. Estimate one fixed unequal-variance parameter from control animals.
4. Use CO/RC to estimate the right-target evidence shift.
5. Transfer that shift to the Blank context to predict P(right | RO).
6. Plot the SDT construction, group means, and individual-animal predictions.

The model is descriptive: it tests whether one shared right-target response
process can account for RO behavior without adding a blindsight-specific
parameter.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, fmean

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "figures"

OUTPUT_STEM = "control_median_sigma_all_experiments"

PERCENT_TOLERANCE = 1.0
EPSILON_RATE = 1e-6

GROUP_ORDER = [
    "Control",
    "V1 ibotenate",
    "HPC ibotenate",
    "Low opacity",
    "V1 muscimol",
    "dLGN muscimol",
    "V1 suction replication",
    "V1 suction",
    "V1 + HPC ibotenate",
    "HPC muscimol",
]

GROUP_AXIS_LABELS = {
    "Control": ("Control",),
    "V1 ibotenate": ("V1", "ibotenate"),
    "HPC ibotenate": ("HPC", "ibotenate"),
    "Low opacity": ("Low", "opacity"),
    "V1 muscimol": ("V1", "muscimol"),
    "dLGN muscimol": ("dLGN", "muscimol"),
    "V1 suction replication": ("V1 suction", "replication"),
    "V1 suction": ("V1", "suction"),
    "V1 + HPC ibotenate": ("V1+HPC", "ibotenate"),
    "HPC muscimol": ("HPC", "muscimol"),
}

GROUP_CLASS = {
    "Control": "Control/reference",
    "V1 ibotenate": "Control/reference",
    "HPC ibotenate": "Control/reference",
    "Low opacity": "Reduced-input controls",
    "V1 muscimol": "Reduced-input controls",
    "dLGN muscimol": "Reduced-input controls",
    "V1 suction replication": "Claimed blindsight-like",
    "V1 suction": "Claimed blindsight-like",
    "V1 + HPC ibotenate": "Claimed blindsight-like",
    "HPC muscimol": "Claimed blindsight-like",
}

CLASS_COLORS = {
    "Control/reference": (90, 90, 90),
    "Reduced-input controls": (0, 114, 178),
    "Claimed blindsight-like": (213, 94, 0),
}

MODEL_COLOR = (0, 158, 115)
OBS_COLOR = (20, 20, 20)
GRID = (230, 230, 230)
AXIS = (150, 150, 150)
NOISE_COLOR = (95, 95, 95)
RC_COLOR = (213, 94, 0)

NORMAL = NormalDist()
SQRT2 = math.sqrt(2.0)
CURRENT_SCALE = 1.0


@dataclass(frozen=True)
class DataSpec:
    source: str
    csv_path: Path
    use_post: bool
    allowed_labels: set[str] | None = None


@dataclass(frozen=True)
class ParsedData:
    rows: list[dict[str, object]]
    exclusions: list[dict[str, object]]


def main_v1_mouse_labels() -> set[str]:
    """Animals from the main V1-suction cohort used for pre/post pairing."""

    return {
        "Mania",
        "Rufus",
        "Visp",
        "Lork",
        "Apple",
        "Taro",
        "Pomey",
        "Velvet",
        "Krys",
        "Worm",
        "Ume",
        "Libra",
        "Eel",
        "Ick",
        "Lime",
        "Sandal",
    }


def data_specs() -> list[DataSpec]:
    main_labels = main_v1_mouse_labels()
    return [
        DataSpec("Control", DATA_DIR / "Fig1E.csv", False, main_labels),
        DataSpec("V1 suction", DATA_DIR / "Fig1E.csv", True, main_labels),
        DataSpec("V1 suction replication", DATA_DIR / "Fig2E.csv", True),
        DataSpec("V1 + HPC ibotenate", DATA_DIR / "Fig 6E.csv", True),
        DataSpec("HPC muscimol", DATA_DIR / "Fig 6P.csv", True),
        DataSpec("V1 ibotenate", DATA_DIR / "Fig 5E.csv", True),
        DataSpec("HPC ibotenate", DATA_DIR / "Fig 6K.csv", True),
        DataSpec("Low opacity", DATA_DIR / "Fig2D.csv", True),
        DataSpec("V1 muscimol", DATA_DIR / "Fig 5J.csv", True),
        DataSpec("dLGN muscimol", DATA_DIR / "Fig 5N.csv", True),
    ]


def norm_sf(x: float) -> float:
    """Standard normal survival function, S(x) = 1 - Phi(x)."""

    return 0.5 * math.erfc(x / SQRT2)


def norm_isf(p: float) -> float:
    """Inverse standard normal survival function."""

    return NORMAL.inv_cdf(1.0 - p)


def safe_norm_isf(p: float) -> float:
    """Inverse survival transform with clipping for exact 0/1 rates."""

    clipped = max(EPSILON_RATE, min(1.0 - EPSILON_RATE, p))
    return norm_isf(clipped)


def adjusted_score(signal_rate: float, baseline_rate: float) -> float:
    denominator = 1.0 - baseline_rate
    if denominator <= 0.0:
        return float("nan")
    return (signal_rate - baseline_rate) / denominator


def z_rate(p: float) -> float:
    """Normal z score for a right-choice rate."""

    return -safe_norm_isf(p)


def implied_sigma(row: dict[str, object]) -> float:
    """Sigma implied when one target shift explains RC/CO and RO/Blank."""

    denominator = z_rate(float(row["ro_right"])) - z_rate(float(row["rc_right"]))
    if abs(denominator) < 1e-12:
        return float("nan")
    return (z_rate(float(row["blank_right"])) - z_rate(float(row["co_right"]))) / denominator


def valid_sigma(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value < 20.0


def estimate_control_sigma(rows: list[dict[str, object]]) -> tuple[float, list[float], int]:
    control_rows = [row for row in rows if str(row.get("source")) == "Control"]
    values = [implied_sigma(row) for row in control_rows]
    valid_values = [value for value in values if valid_sigma(value)]
    if not valid_values:
        raise ValueError("No valid control animals available for sigma estimation.")
    return statistics.median(valid_values), valid_values, len(control_rows)


def rmse(errors: list[float]) -> float:
    finite = [value for value in errors if math.isfinite(value)]
    if not finite:
        return float("nan")
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def mouse_label_from_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    return cleaned.split()[0].strip(",")


def numeric_cells(row: list[str]) -> list[float]:
    values: list[float] = []
    for cell in row:
        stripped = cell.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            continue
    return values


def relevant_triplets(block: list[float]) -> dict[str, tuple[float, float, float]]:
    """Percentage triplets for the four conditions used here.

    Each condition is stored as a three-response block in the extracted CSVs.
    The right-choice columns used by the analysis are indices 4, 7, 13, and 16;
    these triplets are the corresponding full response-rate blocks.
    """

    return {
        "RC": (block[3], block[4], block[5]),
        "CO": (block[6], block[7], block[8]),
        "RO": (block[12], block[13], block[14]),
        "Blank": (block[15], block[16], block[17]),
    }


def triplet_is_extractable(triplet: tuple[float, float, float]) -> bool:
    total = sum(triplet)
    return abs(total - 100.0) <= PERCENT_TOLERANCE


def parse_prepost_rates(spec: DataSpec) -> ParsedData:
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    if not spec.csv_path.exists():
        raise FileNotFoundError(spec.csv_path)

    with spec.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for raw_row in reader:
            if not raw_row or not any(cell.strip() for cell in raw_row):
                continue

            raw_label = raw_row[0].strip()
            label = mouse_label_from_text(raw_label)
            if spec.allowed_labels is not None and label not in spec.allowed_labels:
                continue

            values = numeric_cells(raw_row)
            if len(values) < 36:
                # Prism exports also contain section headers and spacer rows.
                # They are not animal records, so they are ignored rather than
                # counted as analysis exclusions.
                continue

            values = values[-36:]
            block = values[18:36] if spec.use_post else values[:18]
            triplets = relevant_triplets(block)
            bad_triplets = {
                name: triplet
                for name, triplet in triplets.items()
                if not triplet_is_extractable(triplet)
            }
            if bad_triplets:
                reason = "; ".join(
                    f"{name} triplet sum={sum(triplet):.1f}"
                    for name, triplet in bad_triplets.items()
                )
                exclusions.append(
                    {
                        "source": spec.source,
                        "label": label,
                        "raw_label": raw_label,
                        "reason": reason,
                    }
                )
                continue

            rows.append(
                {
                    "condition": f"{spec.source}: {raw_label}",
                    "label": label,
                    "source": spec.source,
                    "blank_right": block[16] / 100.0,
                    "ro_right": block[13] / 100.0,
                    "co_right": block[7] / 100.0,
                    "rc_right": block[4] / 100.0,
                }
            )
    return ParsedData(rows, exclusions)


def load_individual_rates() -> ParsedData:
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for spec in data_specs():
        parsed = parse_prepost_rates(spec)
        rows.extend(parsed.rows)
        exclusions.extend(parsed.exclusions)
    return ParsedData(rows, exclusions)


def predict_from_rc_co_and_blank(
    row: dict[str, object],
    sigma: float,
    model_label: str = "control median sigma",
) -> dict[str, object]:
    """Fit the right-target shift from RC/CO and predict RO from Blank."""

    blank = float(row["blank_right"])
    ro = float(row["ro_right"])
    co = float(row["co_right"])
    rc = float(row["rc_right"])

    c_blank = safe_norm_isf(blank)
    c_co = safe_norm_isf(co)
    rc_target_coordinate = safe_norm_isf(rc)
    mu_raw = c_co - sigma * rc_target_coordinate
    mu = max(0.0, mu_raw)

    predicted_ro = norm_sf((c_blank - mu) / sigma)
    predicted_rc = norm_sf((c_co - mu) / sigma)

    source = str(row.get("source", row.get("label", "")))
    label = str(row.get("label", source))

    observed_ro_adjusted = adjusted_score(ro, blank)
    observed_rc_adjusted = adjusted_score(rc, co)
    predicted_ro_adjusted = adjusted_score(predicted_ro, blank)
    predicted_rc_adjusted = adjusted_score(predicted_rc, co)

    return {
        "condition": row.get("condition", ""),
        "label": label,
        "source": source,
        "n": row.get("n", ""),
        "model_label": model_label,
        "sigma": sigma,
        "observed_blank": blank,
        "observed_ro": ro,
        "observed_co": co,
        "observed_rc": rc,
        "c_blank": c_blank,
        "c_co": c_co,
        "criterion_gap_co_minus_blank": c_co - c_blank,
        "rc_target_coordinate": rc_target_coordinate,
        "mu_raw": mu_raw,
        "mu": mu,
        "mu_clipped_to_zero": mu_raw < 0.0,
        "predicted_ro": predicted_ro,
        "predicted_rc": predicted_rc,
        "ro_error": predicted_ro - ro,
        "rc_error": predicted_rc - rc,
        "observed_ro_adjusted": observed_ro_adjusted,
        "observed_rc_adjusted": observed_rc_adjusted,
        "predicted_ro_adjusted": predicted_ro_adjusted,
        "predicted_rc_adjusted": predicted_rc_adjusted,
        "observed_adjusted_ro_minus_rc": observed_ro_adjusted - observed_rc_adjusted,
        "predicted_adjusted_ro_minus_rc": predicted_ro_adjusted - predicted_rc_adjusted,
        "adjusted_score_error": (
            predicted_ro_adjusted
            - predicted_rc_adjusted
            - (observed_ro_adjusted - observed_rc_adjusted)
        ),
    }


def group_mean_rows(individual_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in individual_rows:
        by_source[str(row["source"])].append(row)

    group_rows: list[dict[str, object]] = []
    for source in GROUP_ORDER:
        rows = by_source.get(source, [])
        if not rows:
            continue
        group_rows.append(
            {
                "condition": source,
                "label": source,
                "source": source,
                "n": len(rows),
                "blank_right": fmean(float(row["blank_right"]) for row in rows),
                "ro_right": fmean(float(row["ro_right"]) for row in rows),
                "co_right": fmean(float(row["co_right"]) for row in rows),
                "rc_right": fmean(float(row["rc_right"]) for row in rows),
            }
        )
    return group_rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sx(value: float, x_min: float, x_max: float, px: float, pw: float) -> float:
    return px + (value - x_min) / (x_max - x_min) * pw


def sy(value: float, y_min: float, y_max: float, py: float, ph: float) -> float:
    return py + ph - (value - y_min) / (y_max - y_min) * ph


class ScaledDraw:
    """Draw helper that renders the same layout at normal/high resolution."""

    def __init__(self, draw: ImageDraw.ImageDraw, scale: float) -> None:
        self.draw = draw
        self.scale = scale

    def _scale_xy(self, xy):
        if isinstance(xy, (list, tuple)):
            if xy and isinstance(xy[0], (list, tuple)):
                return [self._scale_xy(item) for item in xy]
            return tuple(value * self.scale for value in xy)
        return xy * self.scale

    def _scale_width(self, width: int | None) -> int | None:
        if width is None:
            return None
        return max(1, int(round(width * self.scale)))

    def line(self, xy, *args, **kwargs) -> None:
        if "width" in kwargs:
            kwargs["width"] = self._scale_width(kwargs["width"])
        self.draw.line(self._scale_xy(xy), *args, **kwargs)

    def text(self, xy, *args, **kwargs) -> None:
        self.draw.text(self._scale_xy(xy), *args, **kwargs)

    def ellipse(self, xy, *args, **kwargs) -> None:
        if "width" in kwargs:
            kwargs["width"] = self._scale_width(kwargs["width"])
        self.draw.ellipse(self._scale_xy(xy), *args, **kwargs)

    def polygon(self, xy, *args, **kwargs) -> None:
        self.draw.polygon(self._scale_xy(xy), *args, **kwargs)

    def textbbox(self, xy, *args, **kwargs):
        bbox = self.draw.textbbox(self._scale_xy(xy), *args, **kwargs)
        return tuple(value / self.scale for value in bbox)


def font(size: int, bold: bool = False, scale: float = 1.0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    scaled_size = max(1, int(round(size * scale)))
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), scaled_size)
    return ImageFont.load_default()


def configure_fonts(scale: float) -> None:
    global FONT, FONT_BOLD, FONT_SMALL, FONT_SMALL_BOLD, FONT_TINY, FONT_TINY_BOLD, FONT_TITLE
    FONT = font(30, scale=scale)
    FONT_BOLD = font(30, True, scale)
    FONT_SMALL = font(24, scale=scale)
    FONT_SMALL_BOLD = font(24, True, scale)
    FONT_TINY = font(20, scale=scale)
    FONT_TINY_BOLD = font(20, True, scale)
    FONT_TITLE = font(36, True, scale)


configure_fonts(CURRENT_SCALE)


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text_body: str,
    fill: tuple[int, int, int],
    fnt: ImageFont.ImageFont,
    anchor: str = "mm",
    spacing: int = 2,
) -> None:
    lines = text_body.split("\n")
    widths = [draw.textbbox((0, 0), line, font=fnt)[2] for line in lines]
    line_height = draw.textbbox((0, 0), "Ag", font=fnt)[3] + spacing
    total_height = line_height * len(lines) - spacing
    x, y = xy
    top = y - total_height / 2 if "m" in anchor else y
    for index, line in enumerate(lines):
        if anchor.startswith("r"):
            lx = x - widths[index]
        elif anchor.endswith("m"):
            lx = x - widths[index] / 2
        else:
            lx = x
        draw.text((lx, top + index * line_height), line, fill=fill, font=fnt)


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text_body: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text_body, font=fnt)
    pad = max(4, int(round(4 * CURRENT_SCALE)))
    width = bbox[2] - bbox[0] + pad * 2
    height = bbox[3] - bbox[1] + pad * 2
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((pad, pad), text_body, fill=fill + (255,), font=fnt)
    rotated = layer.rotate(90, expand=True)
    x, y = xy
    image.alpha_composite(
        rotated,
        (
            int(x * CURRENT_SCALE - rotated.width / 2),
            int(y * CURRENT_SCALE - rotated.height / 2),
        ),
    )


def draw_circle(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: float,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] | None = None,
    width: int = 3,
) -> None:
    fill = fill or (255, 255, 255)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    px: int,
    py: int,
    pw: int,
    ph: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
) -> None:
    for tick in y_ticks:
        ty = sy(tick, y_min, y_max, py, ph)
        draw.line((px, ty, px + pw, ty), fill=GRID, width=2)
        draw.text((px - 14, ty), f"{tick:g}", fill=(0, 0, 0), font=FONT_SMALL, anchor="rm")
    for tick in x_ticks:
        tx = sx(tick, x_min, x_max, px, pw)
        draw.line((tx, py, tx, py + ph), fill=GRID, width=2)
        draw.text((tx, py + ph + 34), f"{tick:g}", fill=(0, 0, 0), font=FONT_SMALL, anchor="mm")
    draw.line((px, py + ph, px + pw, py + ph), fill=AXIS, width=4)
    draw.line((px, py, px, py + ph), fill=AXIS, width=4)
    draw.text((px + pw / 2, py + ph + 78), x_label, fill=(0, 0, 0), font=FONT_BOLD, anchor="mm")
    draw_rotated_text(image, (px - 88, py + ph // 2), y_label, FONT_BOLD, (0, 0, 0))


def normal_pdf(value: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - mean) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def draw_curve(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 5,
) -> None:
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_sdt_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    example: dict[str, object],
    sigma: float,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    draw.text((x, y), "A", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    draw.text((x + 50, y), "Unequal-variance SDT construction", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    px, py, pw, ph = x + 100, y + 118, w - 145, 330
    x_min, x_max = -1.8, 3.8
    y_min, y_max = 0.0, 0.43

    c_blank = float(example["c_blank"])
    c_co = float(example["c_co"])
    mu = float(example["mu"])
    target_mu = mu

    def map_x(value: float) -> float:
        return sx(value, x_min, x_max, px, pw)

    def map_y(value: float) -> float:
        return sy(value, y_min, y_max, py, ph)

    for tick in [-1, 0, 1, 2, 3]:
        tx = map_x(float(tick))
        draw.line((tx, py, tx, py + ph), fill=GRID, width=2)
        draw.text((tx, py + ph + 34), f"{tick:g}", fill=(0, 0, 0), font=FONT_SMALL, anchor="mm")
    for tick in [0.0, 0.2, 0.4]:
        ty = map_y(tick)
        draw.line((px, ty, px + pw, ty), fill=GRID, width=2)
        draw.text((px - 14, ty), f"{tick:g}", fill=(0, 0, 0), font=FONT_SMALL, anchor="rm")

    samples = [x_min + index * (x_max - x_min) / 240 for index in range(241)]
    noise = [(map_x(z), map_y(normal_pdf(z, 0.0, 1.0))) for z in samples]
    target = [(map_x(z), map_y(normal_pdf(z, target_mu, sigma))) for z in samples]

    draw_curve(draw, noise, NOISE_COLOR, 5)
    draw_curve(draw, target, MODEL_COLOR, 5)
    draw.line((px, py + ph, px + pw, py + ph), fill=AXIS, width=4)
    draw.line((px, py, px, py + ph), fill=AXIS, width=4)
    draw.text((px + pw / 2, py + ph + 78), "Internal evidence", fill=(0, 0, 0), font=FONT_BOLD, anchor="mm")
    draw_rotated_text(image, (px - 72, py + ph // 2), "Density", FONT_BOLD, (0, 0, 0))

    for criterion, color, name in [
        (c_blank, OBS_COLOR, "cBlank"),
        (c_co, RC_COLOR, "cCO"),
    ]:
        gx = map_x(criterion)
        draw.line((gx, py + 8, gx, py + ph), fill=color, width=5)
        draw.text((gx + 10, py - 16), name, fill=color, font=FONT_SMALL_BOLD, anchor="la")

    lx = px + 12
    ly = y + 72
    draw.line((lx, ly, lx + 42, ly), fill=NOISE_COLOR, width=5)
    draw.text((lx + 54, ly - 12), "target absent", fill=(0, 0, 0), font=FONT_SMALL)
    draw.line((lx + 280, ly, lx + 322, ly), fill=MODEL_COLOR, width=5)
    draw.text((lx + 334, ly - 18), "target present", fill=(0, 0, 0), font=FONT_SMALL)


def draw_group_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, object]],
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    draw.text((x, y), "B", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    draw.text((x + 50, y), "Group means", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    px, py, pw, ph = x + 95, y + 118, w - 135, 330
    y_min, y_max = 0.45, 1.0

    ordered_rows = sorted(rows, key=lambda row: GROUP_ORDER.index(str(row["label"])))
    x_positions = [
        px + index * pw / max(1, len(ordered_rows) - 1)
        for index in range(len(ordered_rows))
    ]

    for tick in [0.5, 0.75, 1.0]:
        ty = sy(tick, y_min, y_max, py, ph)
        draw.line((px, ty, px + pw, ty), fill=GRID, width=2)
        draw.text((px - 14, ty), f"{tick:g}", fill=(0, 0, 0), font=FONT_SMALL, anchor="rm")
    draw.line((px, py + ph, px + pw, py + ph), fill=AXIS, width=4)
    draw.line((px, py, px, py + ph), fill=AXIS, width=4)
    draw_rotated_text(image, (px - 86, py + ph // 2), "P(right | RO)", FONT_BOLD, (0, 0, 0))

    for gx, row in zip(x_positions, ordered_rows):
        observed = float(row["observed_ro"])
        predicted = float(row["predicted_ro"])
        gy_obs = sy(observed, y_min, y_max, py, ph)
        gy_pred = sy(predicted, y_min, y_max, py, ph)
        color = CLASS_COLORS[GROUP_CLASS[str(row["label"])]]
        draw.line((gx, gy_obs, gx, gy_pred), fill=(175, 175, 175), width=3)
        draw_circle(draw, gx, gy_obs, 8.5, OBS_COLOR, (255, 255, 255), 3)
        draw_circle(draw, gx, gy_pred, 8.5, MODEL_COLOR, MODEL_COLOR, 2)
        draw_multiline(
            draw,
            (gx, py + ph + 52),
            "\n".join(GROUP_AXIS_LABELS[str(row["label"])]),
            color,
            FONT_TINY,
            "mm",
            1,
        )

    legend_x = px + 12
    legend_y = y + 76
    draw_circle(draw, legend_x, legend_y, 7, OBS_COLOR, (255, 255, 255), 3)
    draw.text((legend_x + 21, legend_y - 11), "Observed", fill=(0, 0, 0), font=FONT_SMALL)
    draw_circle(draw, legend_x + 168, legend_y, 7, MODEL_COLOR, MODEL_COLOR, 2)
    draw.text((legend_x + 189, legend_y - 11), "Predicted", fill=(0, 0, 0), font=FONT_SMALL)


def draw_individual_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, object]],
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    draw.text((x, y), "C", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    draw.text((x + 50, y), "Individual animals", fill=(0, 0, 0), font=FONT_TITLE, anchor="la")
    px, py, pw, ph = x + 95, y + 86, w - 365, h - 145
    draw_axes(
        draw,
        image,
        px,
        py,
        pw,
        ph,
        0,
        1,
        0,
        1,
        [0, 0.25, 0.5, 0.75, 1.0],
        [0, 0.25, 0.5, 0.75, 1.0],
        "Observed P(right | RO)",
        "Predicted P(right | RO)",
    )
    draw.line((px, py + ph, px + pw, py), fill=(170, 170, 170), width=2)

    for row in rows:
        observed = float(row["observed_ro"])
        predicted = float(row["predicted_ro"])
        if not math.isfinite(observed) or not math.isfinite(predicted):
            continue
        source = str(row["source"])
        group_class = GROUP_CLASS.get(source, "Control/reference")
        color = CLASS_COLORS[group_class]
        gx = sx(observed, 0, 1, px, pw)
        gy = sy(predicted, 0, 1, py, ph)
        if source == "Control":
            draw_circle(draw, gx, gy, 6.5, MODEL_COLOR, (255, 255, 255), 3)
        else:
            draw_circle(draw, gx, gy, 6.5, color, color, 1)

    legend_items = [
        ("Control/reference", CLASS_COLORS["Control/reference"], True),
        ("Reduced-input controls", CLASS_COLORS["Reduced-input controls"], True),
        ("Claimed blindsight-like", CLASS_COLORS["Claimed blindsight-like"], True),
        ("Control animals", MODEL_COLOR, False),
    ]
    legend_x = px + pw + 38
    legend_y = py + 35
    for index, (label, color, filled) in enumerate(legend_items):
        ly = legend_y + index * 36
        draw_circle(draw, legend_x, ly, 6.5, color, color if filled else (255, 255, 255), 3 if not filled else 1)
        draw.text((legend_x + 22, ly - 13), label, fill=(0, 0, 0), font=FONT_SMALL)


def draw_png(
    path: Path,
    group_rows: list[dict[str, object]],
    individual_rows: list[dict[str, object]],
    sigma: float,
    scale: float = 1.0,
    dpi: int = 300,
) -> None:
    global CURRENT_SCALE
    CURRENT_SCALE = scale
    configure_fonts(scale)

    image = Image.new("RGBA", (int(1760 * scale), int(1260 * scale)), (255, 255, 255, 255))
    draw = ScaledDraw(ImageDraw.Draw(image), scale)
    example = next((row for row in group_rows if str(row["label"]) == "V1 suction"), group_rows[0])
    draw_sdt_panel(image, draw, example, sigma, 46, 42, 725, 530)
    draw_group_panel(image, draw, group_rows, 745, 42, 970, 570)
    draw_individual_panel(image, draw, individual_rows, 94, 655, 1580, 490)

    image.convert("RGB").save(path, quality=95, dpi=(dpi, dpi))
    CURRENT_SCALE = 1.0
    configure_fonts(CURRENT_SCALE)


def annotate_rows(
    rows: list[dict[str, object]],
    plot_level: str,
    control_valid_values: list[float],
    control_total: int,
    exclusions: list[dict[str, object]],
) -> None:
    for row in rows:
        row["plot_level"] = plot_level
        row["control_valid_sigma_n"] = len(control_valid_values)
        row["control_total_n"] = control_total
        row["excluded_record_n"] = len(exclusions)
        row["ordinary_rounding_tolerance_percent"] = PERCENT_TOLERANCE


def build_analysis() -> tuple[
    float,
    list[float],
    int,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    parsed = load_individual_rates()
    sigma, control_values, control_total = estimate_control_sigma(parsed.rows)

    group_rows = [
        predict_from_rc_co_and_blank(row, sigma)
        for row in group_mean_rows(parsed.rows)
    ]
    individual_rows = [
        predict_from_rc_co_and_blank(row, sigma)
        for row in parsed.rows
    ]

    annotate_rows(group_rows, "group", control_values, control_total, parsed.exclusions)
    annotate_rows(individual_rows, "individual", control_values, control_total, parsed.exclusions)
    return sigma, control_values, control_total, group_rows, individual_rows, parsed.exclusions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigma, control_values, control_total, group_rows, individual_rows, exclusions = build_analysis()

    save_csv(OUT_DIR / f"{OUTPUT_STEM}.csv", group_rows + individual_rows)
    save_csv(OUT_DIR / f"{OUTPUT_STEM}_exclusions.csv", exclusions)
    draw_png(OUT_DIR / f"{OUTPUT_STEM}.png", group_rows, individual_rows, sigma)
    draw_png(
        OUT_DIR / f"{OUTPUT_STEM}_highres.png",
        group_rows,
        individual_rows,
        sigma,
        scale=3,
        dpi=600,
    )
    group_errors = [float(row["ro_error"]) for row in group_rows]
    individual_errors = [
        float(row["ro_error"])
        for row in individual_rows
        if math.isfinite(float(row["ro_error"]))
    ]
    group_errors_without_hpc_muscimol = [
        float(row["ro_error"])
        for row in group_rows
        if str(row["label"]) != "HPC muscimol"
    ]

    print(f"control median sigma: {sigma:.3f} ({len(control_values)} / {control_total} valid controls)")
    print(f"excluded animal-condition records: {len(exclusions)}")
    for exclusion in exclusions:
        print(f"  {exclusion['source']} / {exclusion['label']}: {exclusion['reason']}")
    print(f"group RO RMSE all groups: {rmse(group_errors):.3f}")
    print(f"group RO RMSE without HPC muscimol group: {rmse(group_errors_without_hpc_muscimol):.3f}")
    print(
        f"individual RO RMSE all groups: {rmse(individual_errors):.3f} "
        f"({len(individual_errors)} finite animal-condition observations)"
    )
    print("Group prediction errors:")
    for row in sorted(group_rows, key=lambda item: abs(float(item["ro_error"])), reverse=True):
        print(
            f"  {row['label']}: n={row.get('n', '')}, "
            f"observed={float(row['observed_ro']):.3f}, "
            f"predicted={float(row['predicted_ro']):.3f}, "
            f"error={float(row['ro_error']):+.3f}"
        )


if __name__ == "__main__":
    main()
