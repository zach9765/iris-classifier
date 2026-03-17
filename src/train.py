import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Train an Iris Decision Tree Classifier")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size:  {X_test.shape[0]}")

    # Train model
    clf = DecisionTreeClassifier(random_state=args.random_state, max_depth=5)
    clf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion-matrix figure
    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix - Iris Classifier")
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(fig_path)
    print(f"\nConfusion matrix saved to {fig_path}")


if __name__ == "__main__":
    main()
