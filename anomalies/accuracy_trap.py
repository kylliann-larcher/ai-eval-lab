"""Le piège de l'accuracy sur des données déséquilibrées.

On génère un jeu de données où 1 % des points sont des anomalies, puis on
compare trois modèles :
  1. un modèle « idiot » qui répond toujours « normal » ;
  2. une Isolation Forest (non supervisée) ;
  3. un gradient boosting supervisé avec classes pondérées.

Le modèle idiot obtient 99 % d'accuracy et ne détecte aucune anomalie.
Entre les trois, l'accuracy varie de moins de 2 points alors que le rappel
va de 0 % à plus de 80 %. Le rappel, la précision et la matrice de
confusion le montrent tout de suite.

Usage :
    python anomalies/accuracy_trap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
ANOMALY_RATE = 0.01
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_data(n_samples: int = 20_000, seed: int = SEED):
    """Données synthétiques : classe 1 = anomalie (1 % des points)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[1 - ANOMALY_RATE, ANOMALY_RATE],
        class_sep=1.5,
        flip_y=0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run() -> dict:
    X_train, X_test, y_train, y_test = make_data()

    # 1. Toujours « normal »
    dummy = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)

    # 2. Isolation Forest : -1 = anomalie, 1 = normal
    iso = IsolationForest(contamination=ANOMALY_RATE, random_state=SEED).fit(X_train)
    iso_pred = (iso.predict(X_test) == -1).astype(int)

    # 3. Gradient boosting supervisé, classes pondérées
    hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    hgb.fit(X_train, y_train)

    results = {
        "toujours_normal": evaluate(y_test, dummy.predict(X_test)),
        "isolation_forest": evaluate(y_test, iso_pred),
        "gradient_boosting": evaluate(y_test, hgb.predict(X_test)),
    }
    results["_meta"] = {
        "n_test": int(len(y_test)),
        "n_anomalies_test": int(y_test.sum()),
        "seed": SEED,
    }
    return results


def plot(results: dict, path: Path) -> None:
    """Matrices de confusion. La couleur suit le % de chaque ligne (classe réelle),
    sinon les 5 940 « normal » écrasent tout et la ligne « anomalie » est invisible."""
    names = {
        "toujours_normal": "Toujours « normal »",
        "isolation_forest": "Isolation Forest",
        "gradient_boosting": "Gradient boosting",
    }
    labels = ["normal", "anomalie"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (key, title) in zip(axes, names.items()):
        r = results[key]
        cm = np.array(r["confusion_matrix"])
        row_pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if row_pct[i, j] > 0.5 else "#1f2937"
                pct = row_pct[i, j]
                pct_txt = "<1%" if 0 < pct < 0.01 else f"{pct:.0%}"
                ax.text(j, i, f"{cm[i, j]}\n({pct_txt})", ha="center",
                        va="center", color=color, fontsize=12)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("prédit")
        ax.set_ylabel("réel")
        ax.set_title(
            f"{title}\naccuracy {r['accuracy']:.1%} · rappel {r['recall']:.0%}",
            fontsize=11,
        )
    fig.suptitle("Même jeu de test, 1 % d'anomalies", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run()

    meta = results["_meta"]
    print(f"Jeu de test : {meta['n_test']} points, dont {meta['n_anomalies_test']} anomalies\n")
    print(f"{'modèle':<22}{'accuracy':>10}{'précision':>11}{'rappel':>9}{'F1':>7}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(
            f"{name:<22}{r['accuracy']:>10.1%}{r['precision']:>11.1%}"
            f"{r['recall']:>9.1%}{r['f1']:>7.2f}"
        )

    (RESULTS_DIR / "anomalies.json").write_text(json.dumps(results, indent=2))
    plot(results, RESULTS_DIR / "confusion_matrices.png")
    print(f"\nRésultats : {RESULTS_DIR / 'anomalies.json'}")
    print(f"Figure    : {RESULTS_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
