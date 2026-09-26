"""Create noisy versions of WAV files for a small stress test.

This optional utility mixes white noise into normalized audio. It is intentionally simple
so the noise generation is reproducible and easy to describe in the report.
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def add_noise(audio, snr_db):
    power = np.mean(audio ** 2) + 1e-12
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(42).normal(0, np.sqrt(noise_power), size=audio.shape)
    out = audio + noise
    return np.clip(out, -1.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--snr", type=float, default=10.0)
    args = p.parse_args()
    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for wav in src.glob("*.wav"):
        audio, sr = sf.read(wav, always_2d=False)
        out = add_noise(audio.astype(np.float32), args.snr)
        sf.write(dst / wav.name, out, sr)
        print(dst / wav.name)

if __name__ == "__main__":
    main()
