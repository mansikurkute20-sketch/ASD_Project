from scipy.io import arff
import pandas as pd

# Step 1: Load the ARFF file
data, meta = arff.loadarff("Autism-Child-Data.arff")

# Step 2: Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Display first 5 rows
print("First 5 rows:")
print(df.head())

# Step 4: Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Step 5: Display column names
print("\nColumn Names:")
print(df.columns)

# Step 6: Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Step 7: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())