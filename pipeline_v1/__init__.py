"""pipeline_v1 — panel-wise manga colorization pipeline.

Stages per manga page:
  1-2. detect panels (YOLO26n) + extract them numbered in Japanese reading order
  3.    per-panel character detection (OpenRouter google/gemma-4-31b-it)
  4.    per-panel colorization (FLUX.2 Klein 9B base + manga LoRA on the Spark server,
        atlas filtered to the detected characters)
  5.    stitch the colorized panels back onto the original page
"""

__version__ = "0.1.0"
