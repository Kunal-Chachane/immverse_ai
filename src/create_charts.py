from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

INPUT = RESULTS_DIR / "final_comparison.csv"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing: {INPUT}")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    # ---------------------------------------------------------
    # Chart 1: WER comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    x = range(len(df))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        df["clean_mean_wer"],
        width=width,
        label="Clean",
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["noisy_mean_wer"],
        width=width,
        label="Noisy",
    )

    plt.xticks(x, df["model"], rotation=15)
    plt.ylabel("Mean WER")
    plt.title("ASR Model WER: Clean vs Noisy Audio")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "wer_clean_vs_noisy.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Chart 2: Latency comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        df["clean_mean_latency"],
        width=width,
        label="Clean",
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["noisy_mean_latency"],
        width=width,
        label="Noisy",
    )

    plt.xticks(x, df["model"], rotation=15)
    plt.ylabel("Mean Latency (seconds)")
    plt.title("ASR Model Latency: Clean vs Noisy Audio")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "latency_clean_vs_noisy.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Chart 3: WER degradation
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.bar(
        df["model"],
        df["wer_degradation_percent"],
    )

    plt.xticks(rotation=15)
    plt.ylabel("WER Increase (%)")
    plt.title("WER Degradation Under Noise")
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "wer_degradation.png",
        dpi=200,
    )

    plt.close()

    print("=" * 60)
    print("CHART GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {CHARTS_DIR}")
    print()
    print("Created:")

    for file in sorted(CHARTS_DIR.glob("*.png")):
        print(f"  {file.name}")

    print("=" * 60)


if __name__ == "__main__":
    main()