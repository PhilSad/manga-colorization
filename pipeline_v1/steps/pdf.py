"""Pipeline stage 7: PDF export of the colorized (stitched) pages.

The final pipeline stage. Reads every page from `4_stitched/` (filename
sort order = reading order, natural sort so p010 < p002) and packs them into
a single multi-page PDF with one page per manga page, using Pillow's native
PDF writer (`save_all=True` + `append_images`) — no extra dependency.

Writes `6_pdf/<pdf_name>` (default `colorized.pdf`) + `6_pdf/summary.json`
with per-page provenance records (source file record, pdf page index,
dimensions). Pure image processing: no backends, no network.

The standalone offline tool `scripts/make_pdf.py` delegates to
`run_pdf_step`, so a completed run can be re-exported with custom options
(--page filter, --name, --dpi, --output-dir) without re-running the
pipeline.

Memory note: Pillow's `save_all` requires every page image to be open at
once, so a very large run (e.g. a whole volume of full-size pages) loads
all stitched pages into memory simultaneously. The repo's usual "small test
run" (--skip-first 3 --limit 5) is far below that threshold.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import STEP_DIRS, PipelineConfig
from run_context import RunContext, write_json
from selection import page_selected
from util import SUPPORTED_IMAGE_SUFFIXES, file_record


def run_pdf_step(
    ctx: RunContext,
    config: PipelineConfig,
    *,
    output_dir: Path | None = None,
    page_substrings: tuple[str, ...] = (),
) -> dict:
    """Export all stitched pages of a run into one multi-page PDF.

    Pages are taken from `4_stitched/` in filename order (natural sort =
    reading order). Optional `page_substrings` filters by page-stem
    substring (like the standalone script's `--page` filter). When
    `config.only_panels` is set, only those pages are exported.

    Returns the step record: `{"output": file_record(pdf), "output_dir": ...,
    "pages_in_pdf": N, "records": [...]}` — every record carries
    `{"page", "input" (file record), "pdf_page"}`. Raises ValueError when
    there are no stitched pages to export.
    """
    stitched_dir = ctx.run_dir / STEP_DIRS["stitch"]
    if not stitched_dir.is_dir():
        raise ValueError(
            f"no stitched pages to export ({stitched_dir} missing); "
            "run the 'stitch' step first"
        )
    pages = sorted(
        path
        for path in stitched_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if config.only_panels:
        pages = [p for p in pages if page_selected(p.stem, config.only_panels)]
    if page_substrings:
        pages = [p for p in pages if any(s in p.stem for s in page_substrings)]
    if not pages:
        raise ValueError(f"no stitched pages to export in {stitched_dir}")

    out_dir = output_dir or ctx.step_dir("pdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / config.pdf_name

    images: list[Image.Image] = []
    try:
        for path in tqdm(pages, desc="pdf: load pages", unit="page", leave=False):
            image = Image.open(path)
            image.load()
            images.append(image.convert("RGB"))
        images[0].save(
            pdf_path,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=float(config.pdf_dpi),
            title=ctx.run_dir.name,
            creator="pipeline_v1 (Pillow PDF writer)",
        )
    finally:
        for image in images:
            image.close()

    records: list[dict] = []
    for index, path in enumerate(pages):
        records.append({
            "page": path.stem,
            "input": file_record(path),
            "pdf_page": index + 1,
        })

    summary = {
        "run_dir": str(ctx.run_dir),
        "output_dir": str(out_dir),
        "pdf": file_record(pdf_path, mime_type="application/pdf"),
        "pages_in_pdf": len(records),
        "records": records,
    }
    write_json(out_dir / "summary.json", summary)
    return {
        "output": summary["pdf"],
        "output_dir": str(out_dir),
        "pages_in_pdf": len(records),
        "records": records,
        "pdf_dpi": config.pdf_dpi,
    }
