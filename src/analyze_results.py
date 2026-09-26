from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CLEAN_RESULTS = RESULTS_DIR / "benchmark_results.csv"
NOISY_RESULTS = RESULTS_DIR / "noisy_benchmark_results.csv"

CLEAN_SUMMARY = RESULTS_DIR / "summary.csv"
NOISY_SUMMARY = RESULTS_DIR / "noisy_summary.csv"

OUTPUT = RESULTS_DIR / "final_comparison.csv"


def load_results(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = {
        "model",
        "sample_id",
        "wer",
        "latency_sec",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return df


def main():

    print("=" * 70)
    print("ASR BENCHMARK ANALYSIS")
    print("=" * 70)

    clean = load_results(CLEAN_RESULTS)
    noisy = load_results(NOISY_RESULTS)

    print(f"Clean samples : {clean['sample_id'].nunique()}")
    print(f"Noisy samples : {noisy['sample_id'].nunique()}")
    print()

    clean_summary = (
        clean.groupby("model")
        .agg(
            clean_samples=("sample_id", "count"),
            clean_mean_wer=("wer", "mean"),
            clean_mean_latency=("latency_sec", "mean"),
            clean_median_latency=("latency_sec", "median"),
        )
        .reset_index()
    )

    noisy_summary = (
        noisy.groupby("model")
        .agg(
            noisy_samples=("sample_id", "count"),
            noisy_mean_wer=("wer", "mean"),
            noisy_mean_latency=("latency_sec", "mean"),
            noisy_median_latency=("latency_sec", "median"),
        )
        .reset_index()
    )

    comparison = clean_summary.merge(
        noisy_summary,
        on="model",
        how="inner",
    )

    comparison["wer_increase"] = (
        comparison["noisy_mean_wer"]
        - comparison["clean_mean_wer"]
    )

    comparison["wer_degradation_percent"] = (
        comparison["wer_increase"]
        / comparison["clean_mean_wer"].replace(0, pd.NA)
    ) * 100

    comparison["latency_change_sec"] = (
        comparison["noisy_mean_latency"]
        - comparison["clean_mean_latency"]
    )

    comparison["latency_change_percent"] = (
        comparison["latency_change_sec"]
        / comparison["clean_mean_latency"]
    ) * 100

    comparison = comparison.sort_values(
        "noisy_mean_wer"
    )

    comparison.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 70)
    print("CLEAN vs NOISY COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "clean_mean_wer",
        "noisy_mean_wer",
        "wer_increase",
        "clean_mean_latency",
        "noisy_mean_latency",
        "latency_change_percent",
    ]

    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )

    best_clean = comparison.loc[
        comparison["clean_mean_wer"].idxmin()
    ]

    best_noisy = comparison.loc[
        comparison["noisy_mean_wer"].idxmin()
    ]

    fastest_clean = comparison.loc[
        comparison["clean_mean_latency"].idxmin()
    ]

    fastest_noisy = comparison.loc[
        comparison["noisy_mean_latency"].idxmin()
    ]

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(
        f"Best clean WER   : "
        f"{best_clean['model']} "
        f"({best_clean['clean_mean_wer']:.4f})"
    )

    print(
        f"Best noisy WER   : "
        f"{best_noisy['model']} "
        f"({best_noisy['noisy_mean_wer']:.4f})"
    )

    print(
        f"Fastest clean    : "
        f"{fastest_clean['model']} "
        f"({fastest_clean['clean_mean_latency']:.3f}s)"
    )

    print(
        f"Fastest noisy    : "
        f"{fastest_noisy['model']} "
        f"({fastest_noisy['noisy_mean_latency']:.3f}s)"
    )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(OUTPUT)
    print("=" * 70)


if __name__ == "__main__":
    main()