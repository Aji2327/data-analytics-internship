# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Replace 'your_dataset.csv' with your file path
df = pd.read_csv('HackerRank-Developer-Survey-2018-Numeric.HackerRank-Developer-Survey-2018-Numeric.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic information about dataset
print("\nDataset Info:")
print(df.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Visualize distributions for numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Box plots for numerical columns to check for outliers
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()

# Count plots for categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.show()

# Pairplot to check relationships between variables
print("\nPair Plot:")
sns.pairplot(df[num_cols])
plt.show()

# Heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Identifying outliers using the Z-score method
from scipy.stats import zscore

# Calculate Z-scores
z_scores = pd.DataFrame((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
outliers = (z_scores.abs() > 3).sum()

print("\nOutliers per column (Z-score > 3):")
print(outliers)