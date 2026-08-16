"""Verification loop for the colorize step (`--verify-attempts`).

After a panel is colorized, an OpenRouter vision-language model
(`openai/gpt-5.6-luna`, strict json_schema structured output) checks whether
every character has its canonical Frieren palette — sending the colorized
panel, the original monochrome crop, and the same labelled atlas the
colorizer saw as context (reuses `verify_color.ColorVerifier`). On a "not
good" verdict the loop **outputs the fix prompt** (console + per-panel file)
and, while attempts remain, re-colorizes the panel with the fix prompt
appended to the palette instruction, then verifies again.

Every colorization attempt and every verification response is recorded: the
step writes `<panel>.verify.json` (all attempts), an `attempt_<n>` image file
for **every** attempt that is superseded (including attempt 1, preserved as
`attempt_1.png` when a retry wins — so the original bad colorization is never
lost), and `<panel>.fix_prompt.txt` when a fix prompt was produced. The final
attempt is copied to the canonical `<stem><ext>` name the stitch step
expects, so the stitch step is untouched.

Loop outcomes:
- `verified`        the last verification judged the palette canonical
- `mismatch`        attempts exhausted with a "not good" verdict (fix prompt
                    output; panel keeps its last colorization)
- `verifier_error`  the verifier failed (error/unparseable); the panel keeps
                    its latest colorization and the loop stops without
                    burning retries on a broken verifier
- `colorize_error`  a colorization attempt failed; nothing to verify, loop
                    stops (the record carries the error as usual)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colorizer import ColorizeRecord
from region_edit import draw_boxes, region_instruction
from util import file_record
from verify_color import (
    ERROR,
    MISMATCH,
    UNPARSEABLE,
    VERIFIED,
    ColorVerifier,
)

OUTCOME_VERIFIED = "verified"
OUTCOME_MISMATCH = "mismatch"
OUTCOME_VERIFIER_ERROR = "verifier_error"
OUTCOME_COLORIZE_ERROR = "colorize_error"

# Marker prepended to the canonical palette instruction for retry attempts,
# telling the colorizer the fix prompt is authoritative.
FIX_HEADER = (
    "Correction from the verification pass (authoritative; overrides any "
    "conflicting colors above):"
)


@dataclass
class VerifyLoopResult:
    outcome: str
    colorize: ColorizeRecord            # final colorize record (output = canonical file)
    attempts: list[dict[str, Any]]      # per-attempt logs (see run_verify_loop)
    fix_prompt: str = ""                # last fix prompt output ("" if none)
    verify_calls: int = 0
    successful_verify_calls: int = 0
    verify_cost_usd: float = 0.0
    region_edit_calls: int = 0          # bbox mode: gpt-image-2 region edits
    region_edit_cost_usd: float = 0.0

    @property
    def colorization_retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def _apply_fix(palette_instruction: str, fix_prompt: str) -> str:
    """Canonical palette instruction + the verification fix prompt for a
    retry attempt. The fix block is authoritative (appended last)."""
    block = f"{FIX_HEADER}\n{fix_prompt}"
    if palette_instruction:
        return f"{palette_instruction}\n\n{block}"
    return block


def _synthesize_fix(analyse: str) -> str:
    """Fallback when a mismatch verdict carries no fix_prompt: derive a
    corrective instruction from the model's own analysis text."""
    return f"Corrections from the verification pass: {analyse}"


def run_verify_loop(
    colorizer,
    verifier: ColorVerifier,
    panel: Path,
    atlas: Path | None,
    output: Path,
    palette_instruction: str = "",
    max_attempts: int = 2,
    verify_mode: str = "fix-prompt",
    region_editor=None,
) -> VerifyLoopResult:
    """Colorize + verify up to `max_attempts` times.

    Attempt 1 writes directly to `output` (the canonical name the stitch step
    expects); retries write `<stem>.attempt_<n><ext>`. After the loop the
    final successful colorization is copied to `output` (no-op for attempt 1),
    and the returned record's `.output` always points at the canonical file.
    If a retry supersedes attempt 1, attempt 1's image is preserved as
    `<stem>.attempt_1<ext>` before the canonical is overwritten, so every
    attempt (including the original) is kept on disk and in verify.json.

    `max_attempts` semantics: 1 = verify and output the fix prompt only (no
    re-colorization); N >= 2 = up to N colorization attempts with at most
    N-1 retries.

    `verify_mode="bbox"` (full-page mode only, config-validated) switches
    the retry path: a mismatch verdict that carries `regions` triggers a
    **region edit** instead of a full re-colorization — the rejected image is
    boxed with `draw_boxes` (region_edit.py), and `region_editor` (a
    GptImage2RegionEditor) recolors only the boxed regions. A mismatch with
    empty regions (localization recall miss — the probed failure mode) falls
    back to the fix-prompt full re-colorization, so the loop always has a
    retry path. Each retry attempt doc records `retry_backend`
    ("gpt-image-2-region-edit" | "fix-prompt"), the `regions` it consumed,
    and for region edits the boxed-image record + rendered edit prompt + cost.
    """
    attempts: list[dict[str, Any]] = []
    verify_calls = 0
    successful_verify_calls = 0
    verify_cost_usd = 0.0
    region_edit_calls = 0
    region_edit_cost_usd = 0.0
    fix_prompt = ""
    final_record: ColorizeRecord | None = None
    last_ok_record: ColorizeRecord | None = None
    outcome = OUTCOME_MISMATCH  # default when the last verdict is "not good"

    bbox_mode = verify_mode == "bbox"
    previous_output: Path | None = None  # the rejected image a retry edits

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            attempt_output = output
            prompt = palette_instruction
        else:
            attempt_output = output.with_name(
                f"{output.stem}.attempt_{attempt}{output.suffix}"
            )
            if bbox_mode:
                prompt = ""  # set by the retry branch below
            else:
                prompt = _apply_fix(palette_instruction, fix_prompt)

        # Retry branch (attempt > 1): bbox mode edits the boxed regions when
        # the previous verdict localized them; otherwise (and always in
        # fix-prompt mode) re-colorize the whole image with the fix prompt.
        retry_backend: str | None = None
        regions: list[dict[str, Any]] = []
        boxed_path: Path | None = None
        edit_prompt = ""
        edit_cost_usd: float | None = None
        if attempt > 1:
            if bbox_mode:
                regions = list(verdict.regions) if verdict is not None else []
                if regions and region_editor is not None:
                    retry_backend = "gpt-image-2-region-edit"
                    width, height = region_editor.target_size(previous_output)
                    boxed_path = output.with_name(
                        f"{output.stem}.attempt_{attempt - 1}.boxed.png"
                    )
                    draw_boxes(
                        previous_output, regions, boxed_path, size=(width, height)
                    )
                    instruction = region_instruction(regions)
                    edit_prompt = region_editor.render_prompt(
                        width, height, instruction, palette_instruction
                    )
                    prompt = edit_prompt
                else:
                    retry_backend = "fix-prompt"
                    prompt = _apply_fix(palette_instruction, fix_prompt)
            else:
                prompt = _apply_fix(palette_instruction, fix_prompt)

        if attempt > 1 and retry_backend == "gpt-image-2-region-edit":
            record = region_editor.edit(
                boxed_path, atlas, attempt_output,
                region_instruction(regions), palette_instruction,
            )
            if record.est_cost_usd is not None:
                edit_cost_usd = record.est_cost_usd
                region_edit_cost_usd += record.est_cost_usd
            if record.status == "ok":
                region_edit_calls += 1
        else:
            record = colorizer.colorize(
                panel, atlas, attempt_output, palette_instruction=prompt
            )
        final_record = record

        verdict = None
        if record.status == "ok":
            last_ok_record = record
            verdict = verifier.verify(attempt_output, panel, atlas=atlas)
            verify_calls += 1
            if verdict.status == VERIFIED:
                successful_verify_calls += 1
            if verdict.cost_usd is not None:
                verify_cost_usd += verdict.cost_usd

        attempt_doc: dict[str, Any] = {
            "attempt": attempt,
            "prompt_used": prompt,
            "colorize": record.to_dict(
                boxed_path if boxed_path is not None else panel, atlas
            ),
        }
        if verdict is not None:
            attempt_doc["verify"] = verdict.to_dict()
        if bbox_mode and attempt > 1:
            attempt_doc["retry_backend"] = retry_backend
            if regions:
                attempt_doc["regions"] = regions
            if retry_backend == "gpt-image-2-region-edit":
                attempt_doc["boxed_image"] = file_record(boxed_path)
                attempt_doc["edit_prompt"] = edit_prompt
                attempt_doc["edit_cost_usd"] = edit_cost_usd
        attempts.append(attempt_doc)

        if record.status != "ok":
            outcome = OUTCOME_COLORIZE_ERROR
            break
        if verdict is None:  # defensive: an ok record always gets a verdict
            outcome = OUTCOME_VERIFIER_ERROR
            break
        if verdict.status == VERIFIED:
            outcome = OUTCOME_VERIFIED
            print(f"[verify] {panel.name}: palette verified (attempt {attempt})",
                  flush=True)
            break
        if verdict.status in (UNPARSEABLE, ERROR):
            outcome = OUTCOME_VERIFIER_ERROR
            print(
                f"[verify] {panel.name}: verifier {verdict.status} "
                f"(no retry; keeping attempt {attempt})",
                flush=True,
            )
            break

        # verdict.status == MISMATCH: output the fix prompt for the user and,
        # with attempts left, retry — bbox mode via the region editor when
        # regions were localized, otherwise the fix-prompt re-colorization.
        fix_prompt = (verdict.fix_prompt or _synthesize_fix(verdict.analyse)).strip()
        retry_hint = ""
        if bbox_mode:
            retry_hint = (
                f": {len(verdict.regions)} regions → gpt-image-2 region edit"
                if verdict.regions and region_editor is not None
                else ": no regions → fix-prompt re-colorize"
            )
        print(f"[verify] {panel.name}: palette MISMATCH "
              f"(attempt {attempt}){retry_hint}",
              flush=True)
        print(f"[verify] fix prompt:\n{fix_prompt}", flush=True)
        if attempt < max_attempts:
            previous_output = attempt_output
            continue
        outcome = OUTCOME_MISMATCH
        break

    assert final_record is not None  # max_attempts >= 1 always runs one iteration

    # The canonical output file must hold the final successful colorization:
    # attempt 1 already wrote it; a later retry (or a failed final attempt
    # after a successful one) is copied over. Before overwriting, preserve
    # attempt 1's image as `<stem>.attempt_1<ext>` so EVERY attempt is kept
    # on disk — previously the canonical was left as a byte-copy of the last
    # attempt and the original attempt-1 image was lost forever.
    if last_ok_record is not None and last_ok_record.output != output:
        if len(attempts) > 1 and output.exists():
            attempt_1_path = output.with_name(
                f"{output.stem}.attempt_1{output.suffix}"
            )
            shutil.copy2(output, attempt_1_path)
            # Repoint the attempt-1 provenance record at the preserved file;
            # the canonical name now holds a later attempt's image.
            attempts[0]["colorize"]["output"] = file_record(attempt_1_path)
        shutil.copy2(last_ok_record.output, output)
        last_ok_record.output = output
    if last_ok_record is not None:
        final_record = last_ok_record

    return VerifyLoopResult(
        outcome=outcome,
        colorize=final_record,
        attempts=attempts,
        fix_prompt=fix_prompt,
        verify_calls=verify_calls,
        successful_verify_calls=successful_verify_calls,
        verify_cost_usd=round(verify_cost_usd, 8),
        region_edit_calls=region_edit_calls,
        region_edit_cost_usd=round(region_edit_cost_usd, 8),
    )
