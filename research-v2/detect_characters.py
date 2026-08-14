#!/usr/bin/env python3
"""research-v2: character detection on panels with deepghs/manga109_yolo.

Runs the YOLO detector `deepghs/manga109_yolo` (variant `v2023.12.07_x`,
trained on Manga109, classes `body`/`face`/`frame`/`text`) on each panel crop
and keeps only the `body` detections — the character regions. Draws a
bounding box + label per body on the panel image.

Default input is the panel crops under `research-v2/data/panels/<page>/`
(the output of `split_panels.py`); each page's panels are processed and the
annotations mirror the input layout into the run dir.

Per image, writes into the timestamped run dir:
  <run>/<page>/panel_NNNN.json          body detections (box, confidence)
  <run>/<page>/panel_NNNN_annotated.png bboxes drawn on the panel

Usage:
    .venv/bin/python research-v2/detect_characters.py
    .venv/bin/python research-v2/detect_characters.py --conf 0.355
    .venv/bin/python research-v2/detect_characters.py --input-dir PATH
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MODEL_REPO = "deepghs/manga109_yolo"
MODEL_SUBDIR = "v2023.12.07_x"
MODEL_FILENAME = "model.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_SUBDIR}/{MODEL_FILENAME}"
# Model-card F1-optimal operating point for v2023.12.07_x.
DEFAULT_CONF = 0.355
BODY_CLASS = 0  # names: 0=body, 1=face, 2=frame, 3=text

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "manga109_yolo_v2023.12.07_x.pt"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data" / "panels"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

BOX_COLOR = (30, 180, 60)


def _ensure_model_file(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {model_path}", flush=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "manga-colorization/1"})
    with urllib.request.urlopen(request, timeout=600) as response:
        temporary = model_path.with_suffix(model_path.suffix + ".part")
        temporary.write_bytes(response.read())
    temporary.replace(model_path)
    return model_path


def list_panel_images(input_dir: Path) -> list[Path]:
    """Panel images of an input tree (recursive: <page>/panel_NNNN.png), in
    page-then-panel order."""
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    from PIL import ImageFont

    for candidate in (
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def annotate(image: Image.Image, detections: list[dict], font_size: int = 28) -> Image.Image:
    """Draw a green box + 'body conf' label per body detection on a copy."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(font_size)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=4)
        label = f"body {det['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        pad = 4
        draw.rectangle(
            (x1, max(0, y1 - (bottom - top) - 2 * pad), x1 + (right - left) + 2 * pad, y1),
            fill=BOX_COLOR,
        )
        draw.text(
            (x1 + pad, max(0, y1 - (bottom - top) - pad - top)),
            label, fill=(255, 255, 255), font=font,
        )
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character (body) detection on panels with manga109_yolo."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="panel images, recursive (default: research-v2/data/panels)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="parent of the timestamped run dir (default: research-v2/output)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="YOLO weights (downloaded on first use)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="confidence threshold (default: 0.355, model-card F1 point)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="inference size (checkpoint trained at 640)")
    parser.add_argument("--device", default=None,
                        help="torch device for inference (default: auto)")
    parser.add_argument("--font-size", type=int, default=28,
                        help="label font size on the annotated images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_panel_images(args.input_dir)
    if not images:
        raise SystemExit(f"No panel images found under {args.input_dir}")

    model_path = _ensure_model_file(args.model_path)
    from ultralytics import YOLO  # lazy: torch is a heavy dependency

    model = YOLO(str(model_path))
    if model.names[BODY_CLASS] != "body":
        raise SystemExit(f"unexpected class names {model.names}")

    run_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Batch prediction over all panels, preserving input order.
    results = model.predict(
        [str(p) for p in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    page_records: list[dict] = []
    total_bodies = 0
    for path, result in zip(images, results):
        rel = path.relative_to(args.input_dir)
        page_dir = run_dir / rel.parent
        page_dir.mkdir(parents=True, exist_ok=True)

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls) != BODY_CLASS:
                    continue
                detections.append({
                    "box": [int(v) for v in box.xyxy[0].tolist()],
                    "confidence": round(float(box.conf), 4),
                    "label": "body",
                })

        stem = path.stem
        (page_dir / f"{stem}.json").write_text(
            json.dumps(detections, indent=2) + "\n"
        )
        with Image.open(path) as image:
            image = image.convert("RGB")
            annotated = annotate(image, detections, font_size=args.font_size)
            annotated.save(page_dir / f"{stem}_annotated.png")

        total_bodies += len(detections)
        page_records.append({
            "image": str(rel),
            "panel": stem,
            "page": rel.parent.as_posix(),
            "bodies": detections,
        })
        print(f"{rel}: {len(detections)} body{'s' if len(detections) != 1 else ''}", flush=True)

    manifest = {
        "command": "research-v2/detect_characters.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "input_dir": str(args.input_dir),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device or "auto",
            "labels": "body only (model classes: body, face, frame, text)",
        },
        "backend": {
            "model": f"{MODEL_REPO} {MODEL_SUBDIR} ({MODEL_FILENAME}, ultralytics)",
            "license": "AGPL-3.0 (ultralytics)",
            "cost": "self-hosted / local, $0 per call",
        },
        "totals": {"images": len(page_records), "bodies": total_bodies},
        "pages": page_records,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"\n{len(page_records)} images, {total_bodies} body detections "
        f"(conf>={args.conf}) -> {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
