"""
Production-ready ASR Model Benchmark.

Models:
    - Whisper Small
    - Faster-Whisper Small
    - Wav2Vec2 Base 960h

Dataset:
    - Local LibriSpeech WAV files from data/clean_wav
    - Reference transcripts from LibriSpeech metadata

IMPORTANT:
    This benchmark NEVER decodes the Hugging Face Audio feature.
    Audio is always read from local WAV files using soundfile.

Metrics:
    - Word Error Rate (WER)
    - Mean latency
    - Median latency
    - Memory usage
    - Real-Time Factor (RTF)

Example:
    python src/benchmark.py --num-samples 20 --device cpu
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch

from datasets import load_dataset
from faster_whisper import WhisperModel
from jiwer import wer
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CLEAN_AUDIO_DIR = DATA_DIR / "clean_wav"

RESULTS_DIR = ROOT / "results"

DETAIL_RESULTS = RESULTS_DIR / "benchmark_results.csv"
SUMMARY_RESULTS = RESULTS_DIR / "summary.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

WHISPER_MODEL_ID = "openai/whisper-small"

FASTER_WHISPER_MODEL_ID = (
    "Systran/faster-whisper-small"
)

WAV2VEC_MODEL_ID = (
    "facebook/wav2vec2-base-960h"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

HF_DATASET = "openslr/librispeech_asr"
HF_CONFIG = "clean"
HF_SPLIT = "test"

EXPECTED_SAMPLE_RATE = 16000

DEFAULT_NUM_SAMPLES = 20

SUPPORTED_EXTENSIONS = {".wav"}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def normalize_text(text: Any) -> str:
    """
    Normalize text for WER calculation.
    """

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def memory_mb() -> float:
    """
    Return current process RSS memory in MB.
    """

    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info().rss
        / (1024 ** 2)
    )


def reset_memory() -> None:
    """
    Release Python and CUDA memory where possible.
    """

    gc.collect()

    if torch.cuda.is_available():

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def get_device(
    requested_device: str,
) -> str:
    """
    Resolve requested execution device.
    """

    if requested_device == "auto":

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA "
                "is not available."
            )

        return "cuda"

    return "cpu"


def validate_arguments(
    num_samples: int,
) -> None:

    if num_samples < 1:
        raise ValueError(
            "--num-samples must be >= 1."
        )


# ============================================================
# LOCAL AUDIO DISCOVERY
# ============================================================

def discover_audio_files(
    audio_dir: Path,
    num_samples: int,
) -> list[Path]:
    """
    Find local WAV files.

    The benchmark deliberately uses local WAV files
    instead of Hugging Face Audio decoding.
    """

    if not audio_dir.exists():

        raise FileNotFoundError(
            "Audio directory does not exist:\n"
            f"{audio_dir}"
        )

    files = sorted(
        [
            path
            for path in audio_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not files:

        raise RuntimeError(
            "No WAV files found in:\n"
            f"{audio_dir}"
        )

    selected = files[:num_samples]

    print(
        f"Found {len(files)} local WAV files."
    )

    print(
        f"Using {len(selected)} samples."
    )

    return selected


# ============================================================
# REFERENCE TRANSCRIPTS
# ============================================================

def load_reference_transcripts(
    sample_ids: set[str],
) -> dict[str, str]:
    """
    Load LibriSpeech reference transcripts.

    CRITICAL FIX:

    The Hugging Face Audio column is explicitly removed
    BEFORE iteration.

    We then select ONLY metadata columns.

    Therefore datasets never attempts to decode audio
    and torchcodec is not required.
    """

    print()
    print(
        "Loading LibriSpeech reference transcripts..."
    )

    print(
        "Audio decoding disabled; "
        "metadata only."
    )

    if not sample_ids:
        raise RuntimeError(
            "No sample IDs were provided."
        )

    try:

        dataset = load_dataset(
            HF_DATASET,
            HF_CONFIG,
            split=HF_SPLIT,
            streaming=True,
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load LibriSpeech metadata "
            "from Hugging Face.\n\n"
            "Check your internet connection or "
            "Hugging Face availability.\n\n"
            f"Original error: {exc}"
        ) from exc

    # --------------------------------------------------------
    # CRITICAL SECTION
    # --------------------------------------------------------
    #
    # NEVER access:
    #
    #     row["audio"]
    #
    # NEVER iterate the dataset while the Audio feature
    # is still present.
    #
    # Keep only id + text.
    # --------------------------------------------------------

    available_columns = set(
        dataset.column_names
    )

    if "id" not in available_columns:
        raise RuntimeError(
            "LibriSpeech metadata does not contain "
            "the expected 'id' column.\n"
            f"Available columns: "
            f"{sorted(available_columns)}"
        )

    if "text" not in available_columns:
        raise RuntimeError(
            "LibriSpeech metadata does not contain "
            "the expected 'text' column.\n"
            f"Available columns: "
            f"{sorted(available_columns)}"
        )

    # This is intentional.
    #
    # select_columns creates an iterable dataset containing
    # ONLY id and text.
    #
    # The Audio feature no longer exists in the iterable.
    dataset = dataset.select_columns(
        ["id", "text"]
    )

    references: dict[str, str] = {}

    target_count = len(sample_ids)

    try:

        for row in dataset:

            sample_id = str(
                row.get("id", "")
            ).strip()

            if not sample_id:
                continue

            if sample_id in sample_ids:

                references[sample_id] = (
                    normalize_text(
                        row.get("text", "")
                    )
                )

            if len(references) >= target_count:
                break

    except Exception as exc:

        raise RuntimeError(
            "Failed while reading LibriSpeech "
            "metadata.\n\n"
            "The benchmark is configured to read "
            "only id/text and never decode audio.\n\n"
            f"Original error: {exc}"
        ) from exc

    missing = (
        sample_ids
        - set(references.keys())
    )

    if missing:

        raise RuntimeError(
            "Reference transcripts were not found "
            "for these local WAV files:\n\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        f"Loaded {len(references)} "
        "reference transcripts."
    )

    return references


# ============================================================
# LOAD LOCAL AUDIO
# ============================================================

def load_samples(
    num_samples: int,
) -> list[dict[str, Any]]:
    """
    Load local WAV files and their references.

    Audio is decoded exclusively with soundfile.
    """

    audio_files = discover_audio_files(
        CLEAN_AUDIO_DIR,
        num_samples,
    )

    sample_ids = {
        path.stem
        for path in audio_files
    }

    references = load_reference_transcripts(
        sample_ids
    )

    samples: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "Reading local WAV files..."
    )
    print()

    for index, audio_path in enumerate(
        audio_files,
        start=1,
    ):

        try:

            audio, sample_rate = sf.read(
                audio_path,
                dtype="float32",
                always_2d=False,
            )

            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Stereo -> Mono
            # ------------------------------------------------

            if audio.ndim > 1:

                audio = np.mean(
                    audio,
                    axis=1,
                    dtype=np.float32,
                )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if audio.size == 0:

                raise ValueError(
                    "Audio file is empty."
                )

            if not np.isfinite(
                audio
            ).all():

                raise ValueError(
                    "Audio contains NaN or "
                    "infinite values."
                )

            if sample_rate != (
                EXPECTED_SAMPLE_RATE
            ):

                raise ValueError(
                    f"Expected "
                    f"{EXPECTED_SAMPLE_RATE} Hz, "
                    f"found {sample_rate} Hz."
                )

            sample_id = audio_path.stem

            if sample_id not in references:

                raise KeyError(
                    "No transcript found for "
                    f"{sample_id}"
                )

            duration = (
                len(audio)
                / sample_rate
            )

            if duration <= 0:

                raise ValueError(
                    "Audio duration is zero."
                )

            samples.append(
                {
                    "id": sample_id,
                    "audio_path": str(
                        audio_path
                    ),
                    "audio": audio,
                    "sampling_rate": int(
                        sample_rate
                    ),
                    "duration_sec": float(
                        duration
                    ),
                    "reference": references[
                        sample_id
                    ],
                }
            )

            print(
                f"[{index}/{len(audio_files)}] "
                f"OK    "
                f"{audio_path.name} "
                f"({duration:.2f}s, "
                f"{sample_rate} Hz)"
            )

        except Exception as exc:

            print(
                f"[{index}/{len(audio_files)}] "
                f"FAIL  "
                f"{audio_path.name} "
                f"-> {exc}"
            )

    if not samples:

        raise RuntimeError(
            "No valid benchmark samples "
            "could be loaded."
        )

    print()
    print(
        f"Successfully loaded "
        f"{len(samples)} samples."
    )

    return samples


# ============================================================
# WHISPER SMALL
# ============================================================

def benchmark_whisper(
    samples: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:

    reset_memory()

    print()
    print(
        f"Loading {WHISPER_MODEL_ID}..."
    )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    try:

        processor = (
            AutoProcessor.from_pretrained(
                WHISPER_MODEL_ID
            )
        )

        model = (
            AutoModelForSpeechSeq2Seq
            .from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=dtype,
            )
            .to(device)
        )

        model.eval()

    except Exception as exc:

        raise RuntimeError(
            "Failed to load Whisper Small.\n"
            f"Model: {WHISPER_MODEL_ID}\n"
            f"Error: {exc}"
        ) from exc

    rows: list[
        dict[str, Any]
    ] = []

    try:

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            print(
                f"  Whisper Small: "
                f"sample {index}/"
                f"{len(samples)}"
            )

            start = time.perf_counter()

            inputs = processor(
                sample["audio"],
                sampling_rate=sample[
                    "sampling_rate"
                ],
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value
                in inputs.items()
                if hasattr(value, "to")
            }

            with torch.inference_mode():

                generated = model.generate(
                    **inputs,
                    max_new_tokens=256,
                )

            prediction = (
                processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )[0]
            )

            latency = (
                time.perf_counter()
                - start
            )

            prediction = normalize_text(
                prediction
            )

            rows.append(
                {
                    "model": "Whisper Small",
                    "sample_id": sample["id"],
                    "reference": sample[
                        "reference"
                    ],
                    "prediction": prediction,
                    "latency_sec": latency,
                    "audio_duration_sec": sample[
                        "duration_sec"
                    ],
                    "rtf": (
                        latency
                        / sample[
                            "duration_sec"
                        ]
                    ),
                }
            )

    finally:

        del model
        del processor

        reset_memory()

    return rows


# ============================================================
# FASTER WHISPER SMALL
# ============================================================

def benchmark_faster_whisper(
    samples: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:

    reset_memory()

    compute_type = (
        "float16"
        if device == "cuda"
        else "int8"
    )

    print()
    print(
        f"Loading "
        f"{FASTER_WHISPER_MODEL_ID}..."
    )

    try:

        model = WhisperModel(
            FASTER_WHISPER_MODEL_ID,
            device=device,
            compute_type=compute_type,
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load Faster-Whisper.\n"
            f"Model: "
            f"{FASTER_WHISPER_MODEL_ID}\n"
            f"Device: {device}\n"
            f"Compute type: {compute_type}\n"
            f"Error: {exc}"
        ) from exc

    rows: list[
        dict[str, Any]
    ] = []

    try:

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            print(
                f"  Faster-Whisper Small: "
                f"sample {index}/"
                f"{len(samples)}"
            )

            start = time.perf_counter()

            segments, _ = (
                model.transcribe(
                    sample["audio"],
                    language="en",
                    beam_size=1,
                )
            )

            # IMPORTANT:
            # Faster-Whisper returns a lazy generator.
            # Consume it completely before measuring latency.

            prediction = " ".join(
                segment.text
                for segment in segments
            )

            latency = (
                time.perf_counter()
                - start
            )

            prediction = normalize_text(
                prediction
            )

            rows.append(
                {
                    "model":
                        "Faster-Whisper Small",
                    "sample_id":
                        sample["id"],
                    "reference":
                        sample["reference"],
                    "prediction":
                        prediction,
                    "latency_sec":
                        latency,
                    "audio_duration_sec":
                        sample[
                            "duration_sec"
                        ],
                    "rtf":
                        latency
                        / sample[
                            "duration_sec"
                        ],
                }
            )

    finally:

        del model

        reset_memory()

    return rows


# ============================================================
# WAV2VEC2 BASE 960H
# ============================================================

def benchmark_wav2vec2(
    samples: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:

    reset_memory()

    print()
    print(
        f"Loading {WAV2VEC_MODEL_ID}..."
    )

    try:

        processor = (
            Wav2Vec2Processor
            .from_pretrained(
                WAV2VEC_MODEL_ID
            )
        )

        model = (
            Wav2Vec2ForCTC
            .from_pretrained(
                WAV2VEC_MODEL_ID
            )
            .to(device)
        )

        model.eval()

    except Exception as exc:

        raise RuntimeError(
            "Failed to load Wav2Vec2.\n"
            f"Model: {WAV2VEC_MODEL_ID}\n"
            f"Error: {exc}"
        ) from exc

    rows: list[
        dict[str, Any]
    ] = []

    try:

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            print(
                f"  Wav2Vec2 Base 960h: "
                f"sample {index}/"
                f"{len(samples)}"
            )

            start = time.perf_counter()

            inputs = processor(
                sample["audio"],
                sampling_rate=sample[
                    "sampling_rate"
                ],
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key: value.to(device)
                for key, value
                in inputs.items()
                if hasattr(value, "to")
            }

            with torch.inference_mode():

                logits = model(
                    **inputs
                ).logits

            prediction_ids = (
                torch.argmax(
                    logits,
                    dim=-1,
                )
            )

            prediction = (
                processor.batch_decode(
                    prediction_ids
                )[0]
            )

            latency = (
                time.perf_counter()
                - start
            )

            prediction = normalize_text(
                prediction
            )

            rows.append(
                {
                    "model":
                        "Wav2Vec2 Base 960h",
                    "sample_id":
                        sample["id"],
                    "reference":
                        sample["reference"],
                    "prediction":
                        prediction,
                    "latency_sec":
                        latency,
                    "audio_duration_sec":
                        sample[
                            "duration_sec"
                        ],
                    "rtf":
                        latency
                        / sample[
                            "duration_sec"
                        ],
                }
            )

    finally:

        del model
        del processor

        reset_memory()

    return rows


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    rows: list[dict[str, Any]],
    device: str,
    memory_before: float,
    memory_after: float,
) -> list[dict[str, Any]]:

    memory_delta = (
        memory_after
        - memory_before
    )

    for row in rows:

        reference = normalize_text(
            row["reference"]
        )

        prediction = normalize_text(
            row["prediction"]
        )

        row["reference"] = reference

        row["prediction"] = prediction

        row["wer"] = wer(
            reference,
            prediction,
        )

        row[
            "memory_rss_before_mb"
        ] = memory_before

        row[
            "memory_rss_after_mb"
        ] = memory_after

        row[
            "memory_delta_mb"
        ] = memory_delta

        row["device"] = device

    return rows


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        results
        .groupby("model")
        .agg(
            samples=(
                "sample_id",
                "count",
            ),
            mean_wer=(
                "wer",
                "mean",
            ),
            median_wer=(
                "wer",
                "median",
            ),
            mean_latency_sec=(
                "latency_sec",
                "mean",
            ),
            median_latency_sec=(
                "latency_sec",
                "median",
            ),
            mean_rtf=(
                "rtf",
                "mean",
            ),
            mean_memory_delta_mb=(
                "memory_delta_mb",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment() -> None:

    required_modules = [
        ("numpy", np),
        ("pandas", pd),
        ("psutil", psutil),
        ("soundfile", sf),
        ("torch", torch),
    ]

    missing = []

    for name, module in required_modules:

        if module is None:
            missing.append(name)

    if missing:

        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Production-ready ASR benchmark "
            "for local LibriSpeech WAV files."
        )
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Number of local WAV files "
            "to benchmark."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Execution device.",
    )

    args = parser.parse_args()

    validate_arguments(
        args.num_samples
    )

    check_environment()

    device = get_device(
        args.device
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 60)
    print("ASR MODEL BENCHMARK")
    print("=" * 60)

    print(
        f"Project root : {ROOT}"
    )

    print(
        f"Python       : "
        f"{platform.python_version()}"
    )

    print(
        f"PyTorch      : "
        f"{torch.__version__}"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"CPU threads  : "
        f"{torch.get_num_threads()}"
    )

    if device == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Samples      : "
        f"{args.num_samples}"
    )

    print(
        f"Audio dir    : "
        f"{CLEAN_AUDIO_DIR}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    if not CLEAN_AUDIO_DIR.exists():

        raise FileNotFoundError(
            "Clean audio directory does not exist:\n"
            f"{CLEAN_AUDIO_DIR}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    samples = load_samples(
        args.num_samples
    )

    if not samples:

        raise RuntimeError(
            "No benchmark samples loaded."
        )

    # --------------------------------------------------------
    # Benchmark definitions
    # --------------------------------------------------------

    benchmarks = [
        (
            "Whisper Small",
            lambda:
                benchmark_whisper(
                    samples,
                    device,
                ),
        ),
        (
            "Faster-Whisper Small",
            lambda:
                benchmark_faster_whisper(
                    samples,
                    device,
                ),
        ),
        (
            "Wav2Vec2 Base 960h",
            lambda:
                benchmark_wav2vec2(
                    samples,
                    device,
                ),
        ),
    ]

    all_rows: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Run models
    # --------------------------------------------------------

    for (
        label,
        benchmark_function,
    ) in benchmarks:

        print()
        print("-" * 60)
        print(
            f"RUNNING: {label}"
        )
        print("-" * 60)

        reset_memory()

        memory_before = memory_mb()

        total_start = (
            time.perf_counter()
        )

        try:

            rows = benchmark_function()

        except Exception as exc:

            print()
            print(
                f"ERROR while running "
                f"{label}:"
            )
            print(
                str(exc)
            )

            raise

        total_runtime = (
            time.perf_counter()
            - total_start
        )

        memory_after = memory_mb()

        rows = calculate_metrics(
            rows,
            device,
            memory_before,
            memory_after,
        )

        for row in rows:

            row[
                "model_total_runtime_sec"
            ] = total_runtime

        all_rows.extend(rows)

        print()
        print(
            f"Completed {label} "
            f"in {total_runtime:.2f} seconds."
        )

    # --------------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------------

    results = pd.DataFrame(
        all_rows
    )

    if results.empty:

        raise RuntimeError(
            "Benchmark produced no results."
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results.to_csv(
        DETAIL_RESULTS,
        index=False,
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    summary.to_csv(
        SUMMARY_RESULTS,
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"Detailed results : "
        f"{DETAIL_RESULTS}"
    )

    print(
        f"Summary results  : "
        f"{SUMMARY_RESULTS}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Benchmark interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 60)
        print("BENCHMARK FAILED")
        print("=" * 60)
        print(
            f"Error: {exc}"
        )
        print("=" * 60)

        sys.exit(1)