#!/usr/bin/env python3
"""research-v2: character detection on panels with ultralytics YOLOE,
prompted with the chapter-cast reference images (visual prompting).

YOLOE (open-vocabulary YOLO) is conditioned on visual prompts: binary masks
over regions of the *same canvas* it is asked to detect. Since the cast
references are separate images, each panel is composited onto a canvas with
the cast reference thumbnails (one per character, from `data/refs/`) and the
references' bounding boxes are passed as the visual prompts. Detections whose
box center falls inside the panel region are the panel's characters, labeled
with the cast member the prompt belongs to.

Default cast: chapter c001 (Himmel, Frieren, Eisen, Heiter) from
`pipeline_v1/chapter_casts.json`, mapped to `data/refs/<name>_reference.webp`.

Per panel, writes into the timestamped run dir:
  <run>/<page>/panel_NNNN.json          detections: cast name, box, confidence
  <run>/<page>/panel_NNNN_annotated.png bboxes + labels drawn on the panel
  <run>/ref_sheet.png                   the reference sheet used for the run

Usage:
    .venv/bin/python research-v2/detect_characters_yoloe.py
    .venv/bin/python research-v2/detect_characters_yoloe.py --conf 0.1 --imgsz 1024
    .venv/bin/python research-v2/detect_characters_yoloe.py --model yoloe-26s-seg.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CASTS_JSON = Path(__file__).resolve().parents[1] / "pipeline_v1" / "chapter_casts.json"
DEFAULT_REFS_DIR = Path(__file__).resolve().parents[1] / "data" / "refs"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data" / "panels"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Best config from the panel sweep on CPU (yoloe-26l, refs 300, imgsz 1280).
DEFAULT_MODEL = "yoloe-26l-seg.pt"
MODEL_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
DEFAULT_CONF = 0.1
DEFAULT_REF_SIZE = 300
DEFAULT_IMGSZ = 1280
DEFAULT_CAST_KEY = "c001"

PAD = 10
PANEL_BOX_COLOR = (220, 60, 30)  # red-ish: detections
REF_BOX_COLOR = (40, 120, 220)   # blue: reference prompts (never annotated)


def load_cast(cast_key: str) -> list[str]:
    """Cast members for `cast_key` that have a reference image."""
    casts = json.loads(CASTS_JSON.read_text())
    members = casts["casts"][cast_key]["characters"]
    available = {
        p.name.removesuffix("_reference.webp"): p
        for p in DEFAULT_REFS_DIR.glob("*_reference.webp")
    }
    missing = [m for m in members if m.lower() not in available]
    if missing:
        raise SystemExit(f"cast {cast_key}: no reference image for {missing}")
    return [m for m in members if m.lower() in available]


def build_ref_sheet(cast: list[str], ref_size: int) -> tuple[Image.Image, list[list[int]]]:
    """Composite the cast references side by side; returns (sheet, bboxes)."""
    refs = []
    for name in cast:
        im = Image.open(DEFAULT_REFS_DIR / f"{name.lower()}_reference.webp").convert("RGB")
        im.thumbnail((ref_size, ref_size))
        refs.append(im)
    width = sum(r.width for r in refs) + (len(refs) + 1) * PAD
    height = max(r.height for r in refs) + 2 * PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    bboxes = []
    x = PAD
    for i, ref in enumerate(refs):
        sheet.paste(ref, (x, PAD))
        bboxes.append([x, PAD, x + ref.width, PAD + ref.height])
        x += ref.width + PAD
    return sheet, bboxes


def build_canvas(panel: Image.Image, ref_sheet: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Panel on top, reference sheet below; returns (canvas, panel size)."""
    width = max(panel.width, ref_sheet.width)
    height = panel.height + PAD + ref_sheet.height + PAD
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(panel, (0, 0))
    canvas.paste(ref_sheet, (0, panel.height + PAD))
    return canvas, (panel.width, panel.height)


def _ensure_model(model_name: str) -> Path:
    local = MODELS_DIR / model_name
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MODEL_BASE_URL}/{model_name}"
    print(f"downloading {url} -> {local}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "manga-colorization/1"})
    with urllib.request.urlopen(request, timeout=600) as response:
        temporary = local.with_suffix(local.suffix + ".part")
        temporary.write_bytes(response.read())
    temporary.replace(local)
    return local


def _load_font(size: int) -> ImageFont.ImageFont:
    from PIL import ImageFont

    for candidate in (
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def annotate(image: Image.Image, detections: list[dict], font_size: int = 24) -> Image.Image:
    """Draw a box + 'Name conf' label per detection."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(font_size)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        draw.rectangle((x1, y1, x2, y2), outline=PANEL_BOX_COLOR, width=3)
        label = f"{det['label']} {det['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        pad = 4
        draw.rectangle(
            (x1, max(0, y1 - (bottom - top) - 2 * pad), x1 + (right - left) + 2 * pad, y1),
            fill=PANEL_BOX_COLOR,
        )
        draw.text(
            (x1 + pad, max(0, y1 - (bottom - top) - pad - top)),
            label, fill=(255, 255, 255), font=font,
        )
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character detection on panels with YOLOE visual prompts from the cast refs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="panel images, recursive (default: research-v2/data/panels)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="parent of the timestamped run dir")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLOE checkpoint name (downloaded to research-v2/models/)")
    parser.add_argument("--cast-key", default=DEFAULT_CAST_KEY,
                        help="chapter cast to prompt with (default: c001)")
    parser.add_argument("--ref-size", type=int, default=DEFAULT_REF_SIZE,
                        help="max reference thumbnail size (px)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="inference size for the composited canvas")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="detection confidence threshold")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--font-size", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cast = load_cast(args.cast_key)

    images = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not images:
        raise SystemExit(f"No panel images found under {args.input_dir}")

    model_path = _ensure_model(args.model)
    from ultralytics import YOLOE  # lazy: torch is a heavy dependency
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # noqa: PLC0415

    model = YOLOE(str(model_path))

    ref_sheet, ref_bboxes = build_ref_sheet(cast, args.ref_size)
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ref_sheet.save(run_dir / "ref_sheet.png")

    page_records: list[dict] = []
    total_detections = 0
    for path in images:
        rel = path.relative_to(args.input_dir)
        page_dir = run_dir / rel.parent
        page_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as panel:
            panel = panel.convert("RGB")
            canvas, (pw, ph) = build_canvas(panel, ref_sheet)

        results = model.predict(
            canvas,
            visual_prompts={"bboxes": ref_bboxes, "cls": list(range(len(cast)))},
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            predictor=YOLOEVPSegPredictor,
            conf=args.conf,
        )
        result = results[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cx >= pw or cy >= ph:  # outside the panel -> a ref or noise
                    continue
                detections.append({
                    "label": cast[int(box.cls)],
                    "box": [round(v) for v in (x1, y1, x2, y2)],
                    "confidence": round(float(box.conf), 4),
                })

        # NMS merges per class only; dedupe near-identical boxes across classes
        # by keeping the highest-confidence label per (box center, area) cluster.
        detections = _dedupe(detections)

        stem = path.stem
        (page_dir / f"{stem}.json").write_text(
            json.dumps(detections, indent=2) + "\n"
        )
        annotated = annotate(panel, detections, font_size=args.font_size)
        annotated.save(page_dir / f"{stem}_annotated.png")

        total_detections += len(detections)
        page_records.append({
            "image": str(rel),
            "panel": stem,
            "page": rel.parent.as_posix(),
            "detections": detections,
        })
        labels = ", ".join(d["label"] for d in detections) or "-"
        print(f"{rel}: {labels}", flush=True)

    manifest = {
        "command": "research-v2/detect_characters_yoloe.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "input_dir": str(args.input_dir),
            "model": args.model,
            "cast_key": args.cast_key,
            "cast": cast,
            "ref_size": args.ref_size,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": args.device or "auto",
        },
        "backend": {
            "method": "ultralytics YOLOE, visual prompting with cast reference "
                      f"images ({', '.join(cast)}), SAVPE visual prompts",
            "license": "AGPL-3.0 (ultralytics)",
            "cost": "self-hosted / local, $0 per call",
        },
        "totals": {"images": len(page_records), "detections": total_detections},
        "pages": page_records,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"\n{len(page_records)} images, {total_detections} cast detections "
        f"(conf>={args.conf}) -> {run_dir}",
        flush=True,
    )


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def _dedupe(detections: list[dict]) -> list[dict]:
    """Merge near-identical boxes labeled differently (YOLOE assigns one box
    per class; the same region often fires for several cast members). Keep the
    highest-confidence label per region, merging any box with IoU > 0.3 into
    the region it overlaps most."""
    clusters: list[list[dict]] = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        for cluster in clusters:
            if _iou(det["box"], cluster[0]["box"]) > 0.3:
                cluster.append(det)
                break
        else:
            clusters.append([det])
    return [cluster[0] for cluster in clusters]


if __name__ == "__main__":
    main()
