"""
Prepare a reproducible local LibriSpeech WAV dataset.

Downloads only the requested LibriSpeech test samples in streaming mode,
reads the original audio bytes without automatic audio decoding, validates
the audio, converts it to mono when necessary, and stores it as WAV files.

PyTorch and torchcodec are intentionally NOT required by this script.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = ROOT / "data" / "clean_wav"

DATASET_NAME = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "test"

DEFAULT_NUM_SAMPLES = 20

TARGET_SAMPLE_RATE = 16_000


# ============================================================
# AUDIO HELPERS
# ============================================================

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to float32 mono audio and protect against
    invalid numeric values.
    """

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio contains zero samples.")

    # Convert stereo / multi-channel audio to mono.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1:
        raise ValueError(
            f"Unsupported audio shape: {audio.shape}"
        )

    # Remove NaN / infinity values.
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Ensure audio is inside [-1, 1].
    peak = float(np.max(np.abs(audio)))

    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32)


def validate_wav(
    path: Path,
    expected_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[bool, str]:
    """
    Validate an existing WAV file.
    """

    try:
        info = sf.info(path)

        if info.frames <= 0:
            return False, "zero frames"

        if info.samplerate != expected_sample_rate:
            return (
                False,
                f"sample rate {info.samplerate} != "
                f"{expected_sample_rate}",
            )

        if info.channels != 1:
            return (
                False,
                f"{info.channels} channels != mono",
            )

        return True, "valid"

    except Exception as exc:
        return False, str(exc)


def extract_audio_bytes(audio_field: dict) -> bytes:
    """
    Extract raw audio bytes from an undecoded Hugging Face
    Audio feature.
    """

    if not isinstance(audio_field, dict):
        raise TypeError(
            "Unexpected audio field type: "
            f"{type(audio_field)}"
        )

    audio_bytes = audio_field.get("bytes")

    if audio_bytes is None:
        raise RuntimeError(
            "Audio bytes were not provided by the dataset."
        )

    if not isinstance(
        audio_bytes,
        (bytes, bytearray),
    ):
        raise TypeError(
            "Unexpected audio bytes type: "
            f"{type(audio_bytes)}"
        )

    return bytes(audio_bytes)


# ============================================================
# DATASET
# ============================================================

def load_streaming_dataset():
    """
    Load LibriSpeech in streaming mode and explicitly disable
    automatic audio decoding.

    This prevents the datasets library from requiring torchcodec.
    """

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # CRITICAL:
    # Keep audio encoded as raw bytes.
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    return dataset


# ============================================================
# PREPARATION
# ============================================================

def prepare_audio(
    output_dir: Path,
    num_samples: int,
    overwrite: bool = False,
) -> tuple[int, int, int]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PREPARING CLEAN AUDIO")
    print("=" * 60)

    print(f"Project root : {ROOT}")
    print(f"Output       : {output_dir}")
    print(f"Samples      : {num_samples}")
    print(f"Sample rate  : {TARGET_SAMPLE_RATE} Hz")
    print(f"Overwrite    : {overwrite}")

    print("=" * 60)

    print("\nLoading LibriSpeech in streaming mode...")

    dataset = load_streaming_dataset()

    created = 0
    skipped = 0
    failed = 0

    print("\nProcessing samples...\n")

    for index, row in enumerate(
        dataset.take(num_samples),
        start=1,
    ):

        sample_id = str(
            row.get(
                "id",
                f"sample_{index - 1:04d}",
            )
        )

        output_file = (
            output_dir / f"{sample_id}.wav"
        )

        # --------------------------------------------------------
        # Existing file
        # --------------------------------------------------------

        if output_file.exists() and not overwrite:

            valid, reason = validate_wav(
                output_file
            )

            if valid:
                skipped += 1

                print(
                    f"[{index}/{num_samples}] "
                    f"SKIP  {output_file.name}"
                )

                continue

            print(
                f"[{index}/{num_samples}] "
                f"REBUILD {output_file.name} "
                f"({reason})"
            )

        # --------------------------------------------------------
        # Download / decode raw audio
        # --------------------------------------------------------

        try:

            audio_field = row["audio"]

            audio_bytes = extract_audio_bytes(
                audio_field
            )

            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=False,
            )

            audio = normalize_audio(audio)

            # LibriSpeech clean test audio should be 16 kHz.
            if sample_rate != TARGET_SAMPLE_RATE:
                raise ValueError(
                    f"Unexpected sample rate "
                    f"{sample_rate} Hz; expected "
                    f"{TARGET_SAMPLE_RATE} Hz"
                )

            duration = len(audio) / sample_rate

            # ----------------------------------------------------
            # Atomic temporary write
            # ----------------------------------------------------

            temp_file = output_file.with_name(
                output_file.stem + ".tmp.wav"
            )

            sf.write(
                temp_file,
                audio,
                sample_rate,
                subtype="PCM_16",
            )

            # Validate generated WAV.
            valid, reason = validate_wav(
                temp_file
            )

            if not valid:
                temp_file.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    "Generated WAV failed validation: "
                    f"{reason}"
                )

            # Replace final output.
            temp_file.replace(output_file)

            created += 1

            print(
                f"[{index}/{num_samples}] "
                f"OK    {output_file.name} "
                f"({duration:.2f}s, "
                f"{sample_rate} Hz)"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[{index}/{num_samples}] "
                f"FAIL  {sample_id}: {exc}",
                file=sys.stderr,
            )

    # ============================================================
    # FINAL CHECK
    # ============================================================

    available_files = list(
        output_dir.glob("*.wav")
    )

    print("\n")
    print("=" * 60)
    print("AUDIO PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Created   : {created}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Available : {len(available_files)}")
    print(f"Output    : {output_dir}")

    print("=" * 60)

    if failed == 0:
        print(
            "\nSUCCESS: Clean audio dataset is ready."
        )
    else:
        print(
            "\nWARNING: Some samples failed."
        )

    return created, skipped, failed


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare LibriSpeech audio for "
            "ASR benchmarking."
        )
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Number of LibriSpeech test samples "
            f"to prepare. Default: "
            f"{DEFAULT_NUM_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where WAV files "
            "will be stored."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate existing WAV files."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.num_samples <= 0:
        raise SystemExit(
            "ERROR: --num-samples must be greater than 0."
        )

    prepare_audio(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()