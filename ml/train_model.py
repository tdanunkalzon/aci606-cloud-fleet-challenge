"""Compare and evaluate classifiers on disjoint fleets."""
import joblib
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

DATA = Path("data/training.csv")

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Training dataset not found. Run: python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    print(df.head())
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=606,
    )

    train_idx, test_idx = next(
        splitter.split(df[FEATURES], df[TARGET], groups=df["fleet_id"])
    )

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    print("Flotas de entrenamiento:", sorted(train["fleet_id"].unique()))
    print("Flotas de prueba:", sorted(test["fleet_id"].unique()))
    print("\nValores faltantes en el conjunto de entrenamiento:")
    print(train[FEATURES].isna().sum())
    print("\nMediana de las variables por clase en el conjunto de entrenamiento:")
    print(train.groupby(TARGET)[FEATURES].median().round(3).to_string())

    # Separamos las 12 flotas de entrenamiento en 9 de ajuste y 3 de validación.
    validation_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=607,
    )
    fit_idx, val_idx = next(
        validation_split.split(
            train[FEATURES],
            train[TARGET],
            groups=train["fleet_id"],
        )
    )
    fit_data = train.iloc[fit_idx]
    validation = train.iloc[val_idx]

    print("\nClases en ajuste:")
    print(fit_data[TARGET].value_counts())
    print("\nClases en validación:")
    print(validation[TARGET].value_counts())

    models = {
        "Regresión logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=606),
        ),
    }

    labels = ["NORMAL", "WARNING", "CRITICAL"]
    scores = {}

    for name, model in models.items():
        model.fit(fit_data[FEATURES], fit_data[TARGET])
        predictions = model.predict(validation[FEATURES])
        scores[name] = f1_score(
            validation[TARGET],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        print(f"{name} | F1 macro de validación: {scores[name]:.4f}")

    winner_name = max(scores, key=scores.get)
    print(f"Mayor F1 macro en validación: {winner_name}")
    winner = models[winner_name]
    winner.fit(train[FEATURES], train[TARGET])
    test_predictions = winner.predict(test[FEATURES])

    test_score = f1_score(
        test[TARGET],
        test_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(test[TARGET], test_predictions, labels=labels)
    majority_class = train[TARGET].mode().iloc[0]
    baseline_score = f1_score(
        test[TARGET], [majority_class] * len(test), labels=labels,
        average="macro", zero_division=0,
    )
    print(f"Clase mayoritaria ({majority_class}) | F1 macro de prueba: {baseline_score:.4f}")

    print(f"\n{winner_name} | F1 macro de prueba: {test_score:.4f}")
    print("Matriz de confusión (filas reales, columnas predichas):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, model_path)
    print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    main()
