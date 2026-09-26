from pathlib import Path
import argparse
import gc
import time
import platform

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch
from jiwer import wer
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "noisy_wav"
REFERENCE_CSV = ROOT / "results" / "benchmark_results.csv"
OUT = ROOT / "results" / "noisy_benchmark_results.csv"
SUMMARY_OUT = ROOT / "results" / "noisy_summary.csv"


def memory_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def normalize_text(text):
    return " ".join(str(text).lower().strip().split())


def load_samples():
    """
    Load noisy WAV files and recover their references from
    the clean benchmark results.
    """

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Noisy audio directory not found: {AUDIO_DIR}"
        )

    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(
            f"Clean benchmark results not found: {REFERENCE_CSV}"
        )

    reference_df = pd.read_csv(REFERENCE_CSV)

    # References are identical across the three models,
    # so use the first occurrence for each sample.
    references = (
        reference_df[
            ["sample_id", "reference"]
        ]
        .drop_duplicates("sample_id")
        .set_index("sample_id")["reference"]
        .to_dict()
    )

    samples = []

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        raise RuntimeError(
            f"No WAV files found in {AUDIO_DIR}"
        )

    for wav_file in wav_files:

        sample_id = wav_file.stem

        if sample_id not in references:
            raise RuntimeError(
                f"No reference found for {sample_id}"
            )

        audio, sample_rate = sf.read(
            wav_file,
            dtype="float32",
        )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        samples.append(
            {
                "id": sample_id,
                "audio": np.asarray(audio, dtype=np.float32),
                "sampling_rate": int(sample_rate),
                "reference": normalize_text(
                    references[sample_id]
                ),
            }
        )

    return samples


def benchmark_whisper(samples, device, dtype):

    reset_memory()

    model_id = "openai/whisper-small"

    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)

    model.eval()

    rows = []

    for i, sample in enumerate(samples, start=1):

        print(
            f"  Whisper Small: "
            f"sample {i}/{len(samples)}"
        )

        start = time.perf_counter()

        inputs = processor(
            sample["audio"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        prediction = processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        latency = time.perf_counter() - start

        rows.append(
            {
                "model": "Whisper Small",
                "sample_id": sample["id"],
                "reference": sample["reference"],
                "prediction": normalize_text(prediction),
                "latency_sec": latency,
            }
        )

    del model
    del processor
    reset_memory()

    return rows


def benchmark_faster_whisper(samples, device):

    reset_memory()

    compute_type = (
        "float16"
        if device == "cuda"
        else "int8"
    )

    print(
        "Loading "
        "Systran/faster-whisper-small..."
    )

    model = WhisperModel(
        "Systran/faster-whisper-small",
        device=device,
        compute_type=compute_type,
    )

    rows = []

    for i, sample in enumerate(samples, start=1):

        print(
            f"  Faster-Whisper Small: "
            f"sample {i}/{len(samples)}"
        )

        start = time.perf_counter()

        segments, _ = model.transcribe(
            sample["audio"],
            language="en",
            beam_size=1,
        )

        prediction = " ".join(
            segment.text
            for segment in segments
        )

        latency = time.perf_counter() - start

        rows.append(
            {
                "model": "Faster-Whisper Small",
                "sample_id": sample["id"],
                "reference": sample["reference"],
                "prediction": normalize_text(prediction),
                "latency_sec": latency,
            }
        )

    del model
    reset_memory()

    return rows


def benchmark_wav2vec2(samples, device):

    reset_memory()

    model_id = "facebook/wav2vec2-base-960h"

    print(f"Loading {model_id}...")

    processor = Wav2Vec2Processor.from_pretrained(
        model_id
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        model_id
    ).to(device)

    model.eval()

    rows = []

    for i, sample in enumerate(samples, start=1):

        print(
            f"  Wav2Vec2 Base 960h: "
            f"sample {i}/{len(samples)}"
        )

        start = time.perf_counter()

        inputs = processor(
            sample["audio"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            logits = model(**inputs).logits

        prediction_ids = torch.argmax(
            logits,
            dim=-1,
        )

        prediction = processor.batch_decode(
            prediction_ids
        )[0]

        latency = time.perf_counter() - start

        rows.append(
            {
                "model": "Wav2Vec2 Base 960h",
                "sample_id": sample["id"],
                "reference": sample["reference"],
                "prediction": normalize_text(prediction),
                "latency_sec": latency,
            }
        )

    del model
    del processor
    reset_memory()

    return rows


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ASR models on "
            "10 dB noisy LibriSpeech audio."
        )
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    print("=" * 60)
    print("NOISY ASR MODEL BENCHMARK")
    print("=" * 60)

    print(
        f"Python: {platform.python_version()}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"CPU threads: {torch.get_num_threads()}"
    )

    print(
        f"Audio directory: {AUDIO_DIR}"
    )

    samples = load_samples()

    print(
        f"Loaded {len(samples)} noisy samples."
    )

    all_rows = []

    benchmarks = [
        (
            "Whisper Small",
            lambda: benchmark_whisper(
                samples,
                device,
                dtype,
            ),
        ),
        (
            "Faster-Whisper Small",
            lambda: benchmark_faster_whisper(
                samples,
                device,
            ),
        ),
        (
            "Wav2Vec2 Base 960h",
            lambda: benchmark_wav2vec2(
                samples,
                device,
            ),
        ),
    ]

    for label, benchmark_function in benchmarks:

        print()
        print("-" * 60)
        print(f"RUNNING: {label}")
        print("-" * 60)

        reset_memory()

        memory_before = memory_mb()

        total_start = time.perf_counter()

        rows = benchmark_function()

        total_runtime = (
            time.perf_counter()
            - total_start
        )

        memory_after = memory_mb()

        memory_delta = (
            memory_after
            - memory_before
        )

        for row in rows:

            row["wer"] = wer(
                row["reference"],
                row["prediction"],
            )

            row["device"] = device

            row["memory_rss_before_mb"] = (
                memory_before
            )

            row["memory_rss_after_mb"] = (
                memory_after
            )

            row["memory_delta_mb"] = (
                memory_delta
            )

            row["model_total_runtime_sec"] = (
                total_runtime
            )

            row["condition"] = (
                "noisy_10db_snr"
            )

        all_rows.extend(rows)

        print(
            f"Completed {label} "
            f"in {total_runtime:.2f} seconds."
        )

    results = pd.DataFrame(all_rows)

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUT,
        index=False,
    )

    summary = (
        results
        .groupby("model")
        .agg(
            samples=("sample_id", "count"),
            mean_wer=("wer", "mean"),
            median_latency_sec=(
                "latency_sec",
                "median",
            ),
            mean_latency_sec=(
                "latency_sec",
                "mean",
            ),
            mean_memory_delta_mb=(
                "memory_delta_mb",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    print()
    print("=" * 60)
    print("NOISY BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"Detailed results: {OUT}"
    )

    print(
        f"Summary results:  {SUMMARY_OUT}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()