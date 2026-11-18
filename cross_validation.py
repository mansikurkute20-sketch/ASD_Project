import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from prepare_data import X, y


# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# Evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


print("\n5-Fold Cross Validation Results")
print("=" * 70)


results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    accuracy = scores["test_accuracy"].mean()
    precision = scores["test_precision"].mean()
    recall = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"\n{name}")
    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1-Score :  {f1:.4f}")


# Create comparison table
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
)

print("\n\nFinal Comparison Table")
print("=" * 70)
print(results_df.to_string(index=False))