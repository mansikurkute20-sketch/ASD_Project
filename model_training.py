from scipy.io import arff
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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
    confusion_matrix
)


# ------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------

data, meta = arff.loadarff("Autism-Child-Data.arff")
df = pd.DataFrame(data)

print("Dataset loaded successfully!")
print("Original Dataset Shape:", df.shape)


# ------------------------------------------------
# STEP 2: CONVERT BYTES TO TEXT
# ------------------------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )


# ------------------------------------------------
# STEP 3: REPLACE ? WITH MISSING VALUES
# ------------------------------------------------

df = df.replace("?", pd.NA)


# ------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ------------------------------------------------

df["age"] = df["age"].fillna(df["age"].median())

df["ethnicity"] = df["ethnicity"].fillna("Unknown")

df["relation"] = df["relation"].fillna("Unknown")


# ------------------------------------------------
# STEP 5: CONVERT A1-A10 TO NUMERIC
# ------------------------------------------------

score_columns = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score"
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column])


# ------------------------------------------------
# STEP 6: CLEAN CATEGORICAL VALUES
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


# Fix relation value
df["relation"] = df["relation"].replace({
    "self": "Self"
})


# ------------------------------------------------
# STEP 7: DROP UNNECESSARY FEATURES
# ------------------------------------------------

df = df.drop("age_desc", axis=1)

# Result is automatically calculated screening score.
# We do not use it as an input feature.
df = df.drop("result", axis=1)


# ------------------------------------------------
# STEP 8: ENCODE TARGET
# ------------------------------------------------

df["Class/ASD"] = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})


# ------------------------------------------------
# STEP 9: ONE-HOT ENCODING
# ------------------------------------------------

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# ------------------------------------------------
# STEP 10: X AND y
# ------------------------------------------------

X = df.drop("Class/ASD", axis=1)

y = df["Class/ASD"]


print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFeature Matrix Shape:")
print(X.shape)


# ------------------------------------------------
# STEP 11: TRAIN-TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ------------------------------------------------
# STEP 12: FEATURE SCALING
# ------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------
# STEP 13: CREATE MODELS
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
# STEP 14: TRAIN AND EVALUATE MODELS
# ------------------------------------------------

results = []


for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    # Logistic Regression and KNN use scaled data
    if name in ["Logistic Regression", "KNN"]:

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)


    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1

    })


    # Print results
    print("Accuracy :", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall   :", round(recall, 4))

    print("F1-Score :", round(f1, 4))

    print("\nConfusion Matrix:")

    print(cm)


# ------------------------------------------------
# STEP 15: MODEL COMPARISON
# ------------------------------------------------

results_df = pd.DataFrame(results)


print("\n\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(index=False)
)

# ------------------------------------------------
# STEP 16: SAVE ALL 5 MODELS
# ------------------------------------------------

# Save Logistic Regression
joblib.dump(
    models["Logistic Regression"],
    "logistic_regression_model.pkl"
)

# Save KNN
joblib.dump(
    models["KNN"],
    "knn_model.pkl"
)

# Save Naive Bayes
joblib.dump(
    models["Naive Bayes"],
    "naive_bayes_model.pkl"
)

# Save Decision Tree
joblib.dump(
    models["Decision Tree"],
    "decision_tree_model.pkl"
)

# Save Random Forest
joblib.dump(
    models["Random Forest"],
    "random_forest_model.pkl"
)


# ------------------------------------------------
# STEP 17: SAVE SCALER
# ------------------------------------------------

joblib.dump(
    scaler,
    "scaler.pkl"
)


# ------------------------------------------------
# STEP 18: SAVE FEATURE NAMES
# ------------------------------------------------

joblib.dump(
    X.columns.tolist(),
    "feature_names.pkl"
)


# ------------------------------------------------
# FILES SAVED SUCCESSFULLY
# ------------------------------------------------

print("\n========================================")
print("ALL 5 MODELS SAVED SUCCESSFULLY")
print("========================================")

print("1. logistic_regression_model.pkl")
print("2. knn_model.pkl")
print("3. naive_bayes_model.pkl")
print("4. decision_tree_model.pkl")
print("5. random_forest_model.pkl")
print("6. scaler.pkl")
print("7. feature_names.pkl")

print("\nModel training completed successfully!")


