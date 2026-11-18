from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------

data, meta = arff.loadarff("Autism-Child-Data.arff")
df = pd.DataFrame(data)

# Convert bytes to normal text
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

# Replace ? with missing values
df = df.replace("?", pd.NA)

# ------------------------------------------------
# STEP 2: HANDLE MISSING VALUES
# ------------------------------------------------

df["age"] = df["age"].fillna(df["age"].median())

df["ethnicity"] = df["ethnicity"].fillna("Unknown")
df["relation"] = df["relation"].fillna("Unknown")

# ------------------------------------------------
# STEP 3: CONVERT A1-A10 TO NUMERIC
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

# Fix Self/self inconsistency
df["relation"] = df["relation"].replace({
    "self": "Self"
})

# ------------------------------------------------
# STEP 5: DROP UNNECESSARY / LEAKAGE FEATURES
# ------------------------------------------------

# age_desc is redundant
df = df.drop("age_desc", axis=1)

# result is the automatically calculated screening score
# based on the screening method
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
# STEP 8: SEPARATE FEATURES AND TARGET
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
# DISPLAY RESULTS
# ------------------------------------------------

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nFeature Names:")
print(X.columns.tolist())