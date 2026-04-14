from pathlib import Path

from utils import (
    TARGET_LABELS,
    get_confusion_and_report,
    load_raw_data,
    prepare_dataframe,
    save_artifacts,
    train_and_evaluate_models,
)


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "adult.data"
    artifact_dir = base_dir / "artifacts"

    raw_df = load_raw_data(data_path)
    prepared = prepare_dataframe(raw_df)
    models, results, split_data = train_and_evaluate_models(prepared)

    best_model_name = results.iloc[0]["Model"]
    best_model = models[best_model_name]
    _, X_test, _, y_test = split_data

    cm, report = get_confusion_and_report(best_model, X_test, y_test)
    save_artifacts(best_model, results, artifact_dir)

    print("Dataset shape:", prepared.df.shape)
    print("Best model:", best_model_name)
    print("\nModel comparison:\n", results.to_string(index=False))
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)
    print(f"\nSaved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
