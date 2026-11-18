from scipy.io import arff
import pandas as pd

# 1. Load ARFF dataset
data, meta = arff.loadarff("Autism-Child-Data.arff")

# 2. Convert ARFF data into Pandas DataFrame
df = pd.DataFrame(data)

# 3. Convert bytes values into normal text
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

# 4. Replace '?' with missing values
df = df.replace("?", pd.NA)

# ------------------------------------------------
# STEP 2: HANDLE MISSING VALUES
# ------------------------------------------------

# Fill missing age with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing categorical values with "Unknown"
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
# STEP 4: DROP UNNECESSARY COLUMN
# ------------------------------------------------

df = df.drop("age_desc", axis=1)

# ------------------------------------------------
# STEP 5: ENCODE TARGET COLUMN
# ------------------------------------------------

df["Class/ASD"] = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})

# Check target
print("\nEncoded Target Values:")
print(df["Class/ASD"].value_counts())

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

# Remove extra spaces and make text consistent
for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

# Make relation values consistent
df["relation"] = df["relation"].replace({
    "self": "Self"
})

# ------------------------------------------------
# STEP 7: ENCODE CATEGORICAL FEATURES
# ------------------------------------------------

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# ------------------------------------------------
# CHECK FINAL DATASET
# ------------------------------------------------

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nTarget Distribution:")
print(df["Class/ASD"].value_counts())