#!/usr/bin/env python3
"""
Standalone utility that converts a PDF into a dark-mode friendly variant.

The script relies on ImageMagick's `magick` (or legacy `convert`) binary to
invert page colors and produce a new PDF with light text over a dark
background. This yields a quick, offline-friendly way to re-style documents
for low-light reading.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_LIMITS: dict[str, str] = {
    "memory": "1024MiB",
    "map": "1536MiB",
    "disk": "2GiB",
}

AUTO_BATCH_SIZE = 25


class ProgressDisplay:
    def __init__(self, width: int = 30) -> None:
        self.width = width
        self.last_percent = -1.0
        self.active = False

    def update(self, percent: float) -> None:
        if not math.isfinite(percent):
            return
        percent = max(0.0, min(percent, 100.0))
        if percent <= self.last_percent:
            return
        self.last_percent = percent
        self.active = True
        filled = int((percent / 100.0) * self.width)
        filled = max(0, min(filled, self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stderr.write(f"\r[{bar}] {percent:6.2f}%")
        sys.stderr.flush()

    def finish(self, *, completed: bool = True) -> None:
        if not self.active:
            return
        if completed and self.last_percent < 100.0:
            self.update(100.0)
        sys.stderr.write("\n")
        sys.stderr.flush()


def default_thread_limit() -> int:
    cpu_count = os.cpu_count() or 2
    # Use roughly half the available cores but keep within a modest upper bound.
    return max(1, min(4, (cpu_count + 1) // 2))


def parse_limit_option(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "no", "off", "0"}:
        return None
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF into a dark mode variant using ImageMagick."
    )
    parser.add_argument(
        "source",
        help="Path to the source PDF file.",
    )
    parser.add_argument(
        "destination",
        help="Path for the dark mode PDF to create.",
    )
    parser.add_argument(
        "--error-log",
        help=(
            "Optional path to write 1-based page numbers that failed to convert. "
            "Defaults to <destination>.errors.txt when any pages are skipped."
        ),
    )
    parser.add_argument(
        "--density",
        type=int,
        default=150,
        help="Rendering density (DPI) to use when rasterising pages (default: 150).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Gamma correction applied after inversion to keep text legible (default: 0.9).",
    )
    parser.add_argument(
        "--memory-limit",
        default=DEFAULT_LIMITS["memory"],
        help="ImageMagick memory cap passed via -limit memory (e.g. 1024MiB). Use 'none' to disable.",
    )
    parser.add_argument(
        "--map-limit",
        default=DEFAULT_LIMITS["map"],
        help="ImageMagick pixel cache (map) cap via -limit map (e.g. 1536MiB). Use 'none' to disable.",
    )
    parser.add_argument(
        "--disk-limit",
        default=DEFAULT_LIMITS["disk"],
        help="ImageMagick disk usage cap via -limit disk (e.g. 2GiB). Use 'none' to disable.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=default_thread_limit(),
        help="Upper bound on ImageMagick worker threads (default: half logical cores, up to 4).",
    )
    parser.add_argument(
        "--pages-per-batch",
        type=int,
        default=0,
        help=(
            "Process the PDF in smaller batches of pages to reduce resource usage. "
            "Provide 0 (default) to use the automatic batch size."
        ),
    )
    parser.add_argument(
        "--no-batching",
        action="store_true",
        help="Convert the entire document in one pass without splitting into batches.",
    )
    return parser.parse_args()


def resolve_converter() -> str:
    """
    Locate the ImageMagick binary. Prefer `magick`, fall back to `convert`.
    """
    for candidate in ("magick", "convert"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError(
        "ImageMagick is required but neither `magick` nor `convert` was found in PATH."
    )


def ensure_paths(source: str, destination: str) -> tuple[Path, Path]:
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {source_path}")

    if destination_path.suffix.lower() != ".pdf":
        raise ValueError("Destination path must have a .pdf extension.")

    if not destination_path.parent.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)

    return source_path, destination_path


def build_command(
    converter: str,
    source: Path | str,
    destination: Path,
    density: int,
    gamma: float,
    monitor: bool = False,
    resource_limits: dict[str, str] | None = None,
) -> list[str]:
    command = [converter]
    # When using the newer `magick` entrypoint we need to explicitly specify the 'convert' sub-command.
    if Path(converter).name == "magick":
        command.append("convert")
    if monitor:
        command.append("-monitor")
    if resource_limits:
        for limit_type, limit_value in resource_limits.items():
            if limit_value:
                command.extend(["-limit", limit_type, limit_value])
    command.extend(
        [
            "-density",
            str(density),
            str(source),
            "-colorspace",
            "sRGB",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-level",
            "35%,100%",
            "-channel",
            "RGB,Gray",
            "-negate",
            "+channel",
            "-level",
            "5%,95%",
            "-gamma",
            str(gamma),
            str(destination),
        ]
    )
    return command


def build_env(
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
) -> dict[str, str]:
    env = os.environ.copy()
    if resource_limits:
        for limit_type, limit_value in resource_limits.items():
            if limit_value:
                env_key = f"MAGICK_{limit_type.upper()}_LIMIT"
                env[env_key] = limit_value
    if thread_limit and thread_limit > 0:
        env["MAGICK_THREAD_LIMIT"] = str(thread_limit)
        env["OMP_NUM_THREADS"] = str(thread_limit)
    return env


def execute_conversion(
    converter: str,
    source: Path | str,
    destination: Path,
    density: int,
    gamma: float,
    resource_limits: dict[str, str] | None = None,
    thread_limit: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    command = build_command(
        converter,
        source,
        destination,
        density,
        gamma,
        monitor=True,
        resource_limits=resource_limits,
    )
    progress_pattern = re.compile(r"(\d+(?:\.\d+)?)%")
    internal_display = ProgressDisplay() if progress_callback is None else None

    def handle_progress(percent: float) -> None:
        if progress_callback:
            progress_callback(percent)
        elif internal_display:
            internal_display.update(percent)

    env = build_env(resource_limits, thread_limit)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="ignore",
        env=env,
    )

    if progress_callback:
        progress_callback(0.0)
    elif internal_display:
        internal_display.update(0.0)

    stdout_chunks: list[str] = []
    stderr_lines: list[str] = []
    assert process.stderr is not None
    assert process.stdout is not None

    return_code = 0
    success = False
    try:
        while True:
            line = process.stderr.readline()
            if line:
                stderr_lines.append(line)
                match = progress_pattern.search(line)
                if match:
                    try:
                        percent_value = float(match.group(1))
                    except (OverflowError, ValueError):
                        continue
                    if not math.isfinite(percent_value):
                        continue
                    if percent_value < 0.0:
                        percent_value = 0.0
                    elif percent_value > 100.0:
                        percent_value = 100.0
                    handle_progress(percent_value)
            elif process.poll() is not None:
                break

        remaining_stdout = process.stdout.read()
        if remaining_stdout: 
            stdout_chunks.append(remaining_stdout)

        return_code = process.wait()
        success = return_code == 0
    finally:
        if internal_display:
            internal_display.finish(completed=success)

    stderr_text = "".join(stderr_lines)

    if success:
        if "Too much image data" in stderr_text:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                "ImageMagick emitted a 'Too much image data' warning while preparing PNG data."
            )
        return

    stdout = "".join(stdout_chunks).strip()
    stderr = stderr_text.strip()
    message = (
        f"ImageMagick conversion failed with exit code {return_code}.\n"
        f"Command: {' '.join(command)}\n"
        f"STDOUT: {stdout or '<empty>'}\n"
        f"STDERR: {stderr or '<empty>'}"
    )
    raise RuntimeError(message)


def resolve_identify(converter: str) -> list[str]:
    converter_name = Path(converter).name
    if converter_name == "magick":
        return [converter, "identify"]
    identify_path = shutil.which("identify")
    if identify_path:
        return [identify_path]
    raise RuntimeError(
        "Unable to locate ImageMagick's `identify` utility, which is required for batch processing."
    )


def detect_page_count(
    converter: str,
    source: Path,
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
) -> int:
    identify_cmd = resolve_identify(converter)
    command = identify_cmd + ["-ping", "-format", "%n\n", str(source)]
    env = build_env(resource_limits, thread_limit)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "Unable to determine page count for batching. "
            f"`{' '.join(command)}` exited with {result.returncode}. "
            f"STDERR: {stderr or '<empty>'}"
        )
    output = result.stdout.strip()
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "ImageMagick identify returned no usable output while determining page count."
        )
    try:
        return int(lines[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected output while determining page count: {output!r}"
        ) from exc


def merge_pdfs(
    converter: str,
    chunk_paths: list[Path],
    destination: Path,
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
) -> None:
    if not chunk_paths:
        raise RuntimeError("No chunk PDFs were produced for merging.")
    if len(chunk_paths) == 1:
        shutil.move(str(chunk_paths[0]), destination)
        return

    errors: list[str] = []

    pdfunite = shutil.which("pdfunite")
    if pdfunite:
        command = [pdfunite, *map(str, chunk_paths), str(destination)]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode == 0:
            return
        errors.append(
            f"`{' '.join(command)}` failed with exit code {result.returncode}. "
            f"STDERR: {result.stderr.strip() or '<empty>'}"
        )

    gs = shutil.which("gs")
    if gs:
        command = [
            gs,
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={destination}",
            *map(str, chunk_paths),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode == 0:
            return
        errors.append(
            f"`{' '.join(command)}` failed with exit code {result.returncode}. "
            f"STDERR: {result.stderr.strip() or '<empty>'}"
        )

    command = [converter]
    if Path(converter).name == "magick":
        command.append("convert")
    if resource_limits:
        for limit_type, limit_value in resource_limits.items():
            if limit_value:
                command.extend(["-limit", limit_type, limit_value])
    command.extend(map(str, chunk_paths))
    command.append(str(destination))
    env = build_env(resource_limits, thread_limit)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env,
    )
    if result.returncode != 0:
        errors.append(
            f"`{' '.join(command)}` failed with exit code {result.returncode}. "
            f"STDERR: {result.stderr.strip() or '<empty>'}"
        )
        raise RuntimeError(
            "Failed to merge chunk PDFs into the destination file. "
            + " ".join(errors)
        )


def convert_pages_individually(
    converter: str,
    source: Path,
    page_indices: Iterable[int],
    tmpdir: Path,
    density: int,
    gamma: float,
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
    observer: Callable[[float], None] | None,
    error_pages: list[int],
) -> list[Path]:
    page_list = list(page_indices)
    total_pages = len(page_list)
    if total_pages == 0:
        return []

    if observer:
        observer(0.0)

    generated_paths: list[Path] = []
    for index, page in enumerate(page_list):
        page_path = Path(tmpdir) / f"page_{page:06d}.pdf"
        source_spec = f"{source}[{page}]"
        try:
            execute_conversion(
                converter,
                source_spec,
                page_path,
                density,
                gamma,
                resource_limits=resource_limits,
                thread_limit=thread_limit,
            )
            generated_paths.append(page_path)
        except RuntimeError:
            try:
                page_path.unlink(missing_ok=True)
            except OSError:
                pass
            error_pages.append(page + 1)
        finally:
            if observer:
                step_percent = ((index + 1) / total_pages) * 100.0
                observer(step_percent)

    return generated_paths


def convert_pdf_batched(
    converter: str,
    source: Path,
    destination: Path,
    density: int,
    gamma: float,
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
    pages_per_batch: int,
) -> list[int]:
    error_pages: list[int] = []

    if pages_per_batch <= 0:
        raise ValueError("pages_per_batch must be a positive integer for batching.")

    total_pages = detect_page_count(converter, source, resource_limits, thread_limit)
    if total_pages <= pages_per_batch:
        try:
            execute_conversion(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits=resource_limits,
                thread_limit=thread_limit,
            )
            return error_pages
        except RuntimeError:
            fallback_errors = convert_pdf_with_page_skips(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits,
                thread_limit,
            )
            error_pages.extend(fallback_errors)
            return error_pages

    batches = (total_pages + pages_per_batch - 1) // pages_per_batch
    sys.stderr.write(
        f"Processing {total_pages} pages in {batches} batch(es) of up to {pages_per_batch} pages.\n"
    )
    sys.stderr.flush()

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_paths: list[Path] = []
        overall_display = ProgressDisplay()
        completed_pages = 0

        def overall_callback_factory(
            start_page_index: int, pages_in_batch: int
        ) -> Callable[[float], None]:
            def observer(local_percent: float) -> None:
                if not math.isfinite(local_percent):
                    return
                constrained_local = max(0.0, min(local_percent, 100.0))
                try:
                    overall_pages_done = start_page_index + (
                        constrained_local / 100.0
                    ) * pages_in_batch
                    overall_percent = (overall_pages_done / total_pages) * 100.0
                except OverflowError:
                    return
                if not math.isfinite(overall_percent):
                    return
                overall_display.update(overall_percent)

            return observer

        success = False
        try:
            for batch_index, start_page in enumerate(
                range(0, total_pages, pages_per_batch), start=1
            ):
                end_page = min(start_page + pages_per_batch, total_pages) - 1
                human_start = start_page + 1
                human_end = end_page + 1
                pages_in_batch = end_page - start_page + 1
                sys.stderr.write(
                    f"Batch {batch_index}/{batches}: pages {human_start}-{human_end}\n"
                )
                sys.stderr.flush()
                chunk_path = Path(tmpdir) / f"chunk_{batch_index:04d}.pdf"
                source_spec = f"{source}[{start_page}-{end_page}]"
                batch_observer = overall_callback_factory(
                    completed_pages, pages_in_batch
                )
                try:
                    execute_conversion(
                        converter,
                        source_spec,
                        chunk_path,
                        density,
                        gamma,
                        resource_limits=resource_limits,
                        thread_limit=thread_limit,
                        progress_callback=batch_observer,
                    )
                    chunk_paths.append(chunk_path)
                except RuntimeError:
                    try:
                        chunk_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    page_indices = range(start_page, end_page + 1)
                    per_page_paths = convert_pages_individually(
                        converter,
                        source,
                        page_indices,
                        Path(tmpdir),
                        density,
                        gamma,
                        resource_limits,
                        thread_limit,
                        batch_observer,
                        error_pages,
                    )
                    chunk_paths.extend(per_page_paths)
                completed_pages += pages_in_batch
            success = bool(chunk_paths)
        finally:
            overall_display.finish(completed=success)

        if not chunk_paths:
            raise RuntimeError("All pages failed to convert; no output produced.")

        merged_path = Path(tmpdir) / "merged.pdf"
        merge_pdfs(
            converter,
            chunk_paths,
            merged_path,
            resource_limits=resource_limits,
            thread_limit=thread_limit,
        )
        shutil.move(str(merged_path), destination)

    return error_pages


def convert_pdf_with_page_skips(
    converter: str,
    source: Path,
    destination: Path,
    density: int,
    gamma: float,
    resource_limits: dict[str, str] | None,
    thread_limit: int | None,
) -> list[int]:
    total_pages = detect_page_count(converter, source, resource_limits, thread_limit)
    if total_pages <= 0:
        raise RuntimeError("No pages detected in source PDF; nothing to convert.")

    sys.stderr.write(
        "Encountered conversion errors; retrying per-page with skip-on-error enabled.\n"
    )
    sys.stderr.flush()

    error_pages: list[int] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        display = ProgressDisplay()
        page_paths = convert_pages_individually(
            converter,
            source,
            range(total_pages),
            Path(tmpdir),
            density,
            gamma,
            resource_limits,
            thread_limit,
            display.update,
            error_pages,
        )
        display.finish(completed=bool(page_paths))

        if not page_paths:
            raise RuntimeError("All pages failed to convert; no output produced.")

        merged_path = Path(tmpdir) / "merged.pdf"
        merge_pdfs(
            converter,
            page_paths,
            merged_path,
            resource_limits=resource_limits,
            thread_limit=thread_limit,
        )
        shutil.move(str(merged_path), destination)

    return error_pages


def convert_pdf_with_ghostscript(
    source: Path,
    destination: Path,
    gamma: float,
) -> None:
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError(
            "ImageMagick produced invalid output and Ghostscript fallback is unavailable (gs not found)."
        )

    gamma_value = gamma if gamma and math.isfinite(gamma) and gamma > 0 else 1.0
    transfer_proc = "{1 exch sub dup 0 le {pop 0} {dup ln %.8g mul exp} ifelse}" % (
        gamma_value,
    )

    command = [
        gs,
        "-o",
        str(destination),
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dAutoRotatePages=/None",
        "-dProcessColorModel=/DeviceRGB",
        "-dColorConversionStrategy=/sRGB",
        "-dOverrideICC",
        "-c",
        transfer_proc,
        "-f",
        str(source),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        raise RuntimeError(
            "Ghostscript dark mode conversion failed. "
            f"Command: {' '.join(command)} "
            f"STDOUT: {stdout or '<empty>'} "
            f"STDERR: {stderr or '<empty>'}"
        )


def convert_pdf(
    source: Path,
    destination: Path,
    density: int,
    gamma: float,
    resource_limits: dict[str, str] | None = None,
    thread_limit: int | None = None,
    pages_per_batch: int | None = None,
) -> list[int]:
    converter = resolve_converter()
    using_batches = pages_per_batch is not None
    auto_batches = False
    effective_batch_size: int | None
    if using_batches:
        if pages_per_batch and pages_per_batch > 0:
            effective_batch_size = pages_per_batch
        else:
            effective_batch_size = AUTO_BATCH_SIZE
            auto_batches = True
    else:
        effective_batch_size = None

    error_pages: list[int] = []
    try:
        if using_batches and effective_batch_size is not None:
            if auto_batches:
                sys.stderr.write(
                    f"Using automatic batching with up to {effective_batch_size} page(s) per job.\n"
                )
                sys.stderr.flush()
            error_pages = convert_pdf_batched(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits=resource_limits,
                thread_limit=thread_limit,
                pages_per_batch=effective_batch_size,
            )
        else:
            execute_conversion(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits=resource_limits,
                thread_limit=thread_limit,
            )
            error_pages = []
    except RuntimeError as exc:
        message_lower = str(exc).lower()
        cache_exhausted = "cache resources exhausted" in message_lower
        png_overflow = "too much image data" in message_lower
        if cache_exhausted and not using_batches:
            sys.stderr.write(
                "ImageMagick reported cache exhaustion; retrying with automatic page batching.\n"
            )
            sys.stderr.flush()
            error_pages = convert_pdf_batched(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits=resource_limits,
                thread_limit=thread_limit,
                pages_per_batch=AUTO_BATCH_SIZE,
            )
        elif png_overflow:
            sys.stderr.write(
                "ImageMagick signalled corrupted raster output; falling back to Ghostscript inversion.\n"
            )
            sys.stderr.flush()
            convert_pdf_with_ghostscript(
                source,
                destination,
                gamma,
            )
            error_pages = []
        else:
            error_pages = convert_pdf_with_page_skips(
                converter,
                source,
                destination,
                density,
                gamma,
                resource_limits,
                thread_limit,
            )
    return error_pages


def main() -> None:
    args = parse_args()
    try:
        source_path, destination_path = ensure_paths(args.source, args.destination)
        requested_limits = {
            key: parse_limit_option(getattr(args, f"{key}_limit"))
            for key in ("memory", "map", "disk")
        }
        resource_limits = (
            {
                key: value
                for key, value in requested_limits.items()
                if value
            }
            or None
        )
        thread_limit = max(1, args.threads) if args.threads else None
        if args.no_batching and args.pages_per_batch > 0:
            raise ValueError(
                "--no-batching cannot be combined with a positive --pages-per-batch value."
            )
        if not args.no_batching and args.pages_per_batch < 0:
            raise ValueError("--pages-per-batch must be zero or a positive integer.")
        effective_pages_per_batch = None if args.no_batching else args.pages_per_batch
        error_pages = convert_pdf(
            source_path,
            destination_path,
            args.density,
            args.gamma,
            resource_limits=resource_limits,
            thread_limit=thread_limit,
            pages_per_batch=effective_pages_per_batch,
        )
        error_log_path = (
            Path(args.error_log).expanduser()
            if args.error_log
            else destination_path.with_suffix(".errors.txt")
        )
        if error_pages:
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            error_log_path.write_text(
                "\n".join(str(index) for index in sorted(error_pages)) + "\n",
                encoding="utf-8",
            )
            sys.stderr.write(
                f"Skipped {len(error_pages)} page(s); details written to {error_log_path}.\n"
            )
            sys.stderr.flush()
        else:
            try:
                error_log_path.unlink()
            except FileNotFoundError:
                pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
