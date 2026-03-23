# Iris Classifier (Decision Tree)

## Overview

End-to-end ML example from **Digital Marketing Mastery Module** — builds a decision-tree classifier on the classic Iris dataset using scikit-learn.

## Quick Start

```bash
git clone https://github.com/iris-classifier.git
cd iris-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

CLI options:

```bash
python src/train.py --test-size 0.2 --random-state 42
```

The script prints accuracy to the console and saves the confusion-matrix heatmap to `outputs/confusion_matrix.png`.

## Project Structure

```
iris-classifier/
├── src/
│   └── train.py              # CLI training script
├── outputs/                   # Generated at runtime
│   ├── model.joblib
│   └── confusion_matrix.png
├── notebooks/
│   └── iris_classifier.ipynb  # Jupyter notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Tests

Optional tests can be added under `tests/` — for example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def test_accuracy_minimum():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    assert accuracy >= 0.9, f"Accuracy {accuracy:.2f} is below 0.9 threshold"
```

Run with:

```bash
pip install pytest
pytest tests/
```
