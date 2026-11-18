import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from scipy.io import arff


# ------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------

data, meta = arff.loadarff("Autism-Child-Data.arff")
df = pd.DataFrame(data)

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

df = df.replace("?", pd.NA)


# ------------------------------------------------
# STEP 2: HANDLE MISSING VALUES
# ------------------------------------------------

df["age"] = df["age"].fillna(df["age"].median())
df["ethnicity"] = df["ethnicity"].fillna("Unknown")
df["relation"] = df["relation"].fillna("Unknown")


# ------------------------------------------------
# STEP 3: CONVERT A1-A10
# ------------------------------------------------

score_columns = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score"
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column])


# ------------------------------------------------
# STEP 4: CLEAN CATEGORICAL VALUES
# ------------------------------------------------

categorical_columns = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation"
]

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

df["relation"] = df["relation"].replace({
    "self": "Self"
})


# ------------------------------------------------
# STEP 5: DROP UNNECESSARY FEATURES
# ------------------------------------------------

df = df.drop("age_desc", axis=1)
df = df.drop("result", axis=1)


# ------------------------------------------------
# STEP 6: ENCODE TARGET
# ------------------------------------------------

df["Class/ASD"] = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})


# ------------------------------------------------
# STEP 7: ONE-HOT ENCODING
# ------------------------------------------------

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# ------------------------------------------------
# STEP 8: X AND y
# ------------------------------------------------

X = df.drop("Class/ASD", axis=1)
y = df["Class/ASD"]


# ------------------------------------------------
# STEP 9: TRAIN-TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# STEP 10: SCALING
# ------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------
# STEP 11: MODELS
# ------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ------------------------------------------------
# STEP 12: MODEL EVALUATION
# ------------------------------------------------

results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # ------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\n================================")
    print(name)
    print("================================")

    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NO", "YES"]
    )

    disp.plot()
    plt.title(name + " - Confusion Matrix")
    plt.show()


# ------------------------------------------------
# STEP 13: RESULTS TABLE
# ------------------------------------------------

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

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(results_df.to_string(index=False))


# ------------------------------------------------
# STEP 14: MODEL COMPARISON GRAPH
# ------------------------------------------------

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("ML Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ------------------------------------------------
# STEP 15: CROSS-VALIDATION
# ------------------------------------------------

print("\n================================")
print("5-FOLD CROSS-VALIDATION")
print("================================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

    else:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

    print("\n" + name)
    print("Fold Accuracies:",
          [round(score, 4) for score in scores])

    print(
        "Mean Accuracy:",
        round(scores.mean(), 4)
    )

    print(
        "Standard Deviation:",
        round(scores.std(), 4)
    )