"""Write the classifier example: test predictions of a small digit classifier.

The model is a deliberately weak logistic regression on scikit-learn's
bundled digits data, so the saved predictions contain errors to route.
"""

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples" / "digit_classifier.jsonl"
    )
    output = parser.parse_args().output
    images, labels = load_digits(return_X_y=True)
    train_x, test_x, train_y, test_y, _, test_ids = train_test_split(
        images, labels, range(len(labels)), test_size=0.6, random_state=7
    )
    model = LogisticRegression(max_iter=60, C=0.002).fit(train_x, train_y)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for item, probabilities, truth in zip(
            test_ids, model.predict_proba(test_x), test_y, strict=True
        ):
            top, second = sorted(probabilities, reverse=True)[:2]
            record = {
                "image_id": f"img_{item:04d}",
                "predicted": int(probabilities.argmax()),
                "actual": int(truth),
                "top_prob": round(float(top), 6),
                "margin": round(float(top - second), 6),
            }
            target.write(json.dumps(record) + "\n")
    print(f"Wrote {len(test_y)} predictions to {output}")


if __name__ == "__main__":
    main()
