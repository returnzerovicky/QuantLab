"""
QuantLab AI
Benchmark Different Models
"""

import pandas as pd

from .train import Trainer
from .evaluate import evaluate


MODELS = [
    "lstm",
    "gru",
    "transformer"
]


def benchmark():

    results = []

    print("=" * 70)
    print("Running Model Benchmark")
    print("=" * 70)

    for model_name in MODELS:

        print(f"\nTraining {model_name.upper()}...\n")

        trainer = Trainer(model_name=model_name)

        trainer.fit()

        metrics = evaluate(model_name=model_name)

        metrics["Model"] = model_name.upper()

        results.append(metrics)

    df = pd.DataFrame(results)

    df = df[
        [
            "Model",
            "RMSE",
            "MAE",
            "MAPE",
            "R2"
        ]
    ]

    print("\nBenchmark Results\n")

    print(df)

    df.to_csv(
        "experiments/benchmark_results.csv",
        index=False
    )

    print(
        "\nResults saved to experiments/benchmark_results.csv"
    )


if __name__ == "__main__":

    benchmark()
