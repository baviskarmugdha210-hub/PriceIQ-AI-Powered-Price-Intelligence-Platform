
import pandas as pd
import os

# Dataset path
file_path = "data/raw/vehicles.csv"

# Check file size
file_size = os.path.getsize(file_path) / (1024 ** 3)
print(f"Dataset size: {file_size:.2f} GB")

# Read only the first 5 rows
df = pd.read_csv(file_path, nrows=5)

# Display basic information
print("\nFIRST 5 ROWS")
print(df.to_string())

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nNUMBER OF COLUMNS")
print(len(df.columns))

# Read a larger sample for initial quality analysis
df = pd.read_csv(file_path, nrows=100000)

print("\nDATASET SHAPE (SAMPLE)")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().mean() * 100).round(2)
})
print(missing.sort_values("Missing Percentage", ascending=False))

print("\nDUPLICATE RECORDS IN SAMPLE")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY")
print(df.describe().T)

# Dataset quality report
import os

os.makedirs("reports", exist_ok=True)

# Missing value analysis
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Count": df.isnull().sum().values,
    "Missing Percentage": (df.isnull().mean() * 100).round(2).values
})

missing_report = missing_report.sort_values(
    "Missing Percentage", ascending=False
)

missing_report.to_csv("reports/missing_values_report.csv", index=False)

# Numerical statistics
df.describe().T.to_csv("reports/numerical_summary.csv")

# Dataset overview
with open("reports/dataset_overview.txt", "w") as file:
    file.write(f"Sample rows: {len(df)}\n")
    file.write(f"Columns: {len(df.columns)}\n")
    file.write(f"Duplicate rows in sample: {df.duplicated().sum()}\n")

print("EDA reports generated successfully!")

important_columns = [
    "price",
    "year",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state"
]

for col in important_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False).head(5))

import matplotlib.pyplot as plt
import os

os.makedirs("reports/figures", exist_ok=True)

# Remove invalid prices for visualization
price_df = df[
    (df["price"] > 0) &
    (df["price"] < df["price"].quantile(0.99))
].copy()

# 1. Vehicle price distribution
plt.figure(figsize=(9, 5))
plt.hist(price_df["price"], bins=50)
plt.title("Distribution of Vehicle Prices")
plt.xlabel("Price ($)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.savefig("reports/figures/price_distribution.png")
plt.close()

# 2. Vehicle manufacturing year distribution
if "year" in df.columns:
    plt.figure(figsize=(9, 5))
    df["year"].dropna().hist(bins=30)
    plt.title("Distribution of Vehicle Manufacturing Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Vehicles")
    plt.tight_layout()
    plt.savefig("reports/figures/year_distribution.png")
    plt.close()

# 3. Price versus odometer reading
if "odometer" in df.columns:
    plot_df = price_df.dropna(subset=["odometer"])
    plot_df = plot_df[plot_df["odometer"] < plot_df["odometer"].quantile(0.99)]

    plt.figure(figsize=(9, 5))
    plt.scatter(plot_df["odometer"], plot_df["price"], alpha=0.3)
    plt.title("Vehicle Price vs Odometer")
    plt.xlabel("Odometer Reading")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("reports/figures/price_vs_odometer.png")
    plt.close()

print("Visualizations generated successfully!")

# Outlier and invalid-value analysis

print("\n--- PRICE ANALYSIS ---")
print(df["price"].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]))

print("\nInvalid or zero prices:", (df["price"] <= 0).sum())

print("\n--- YEAR ANALYSIS ---")
print(df["year"].describe())
print("Missing years:", df["year"].isnull().sum())
print("Years before 1980:", (df["year"] < 1980).sum())
print("Years after 2026:", (df["year"] > 2026).sum())

print("\n--- ODOMETER ANALYSIS ---")
print(df["odometer"].describe())
print("Negative odometer readings:", (df["odometer"] < 0).sum())
print("Odometer above 500,000:", (df["odometer"] > 500000).sum())

# Save numerical outlier summary
outlier_report = pd.DataFrame({
    "Check": [
        "Zero or negative prices",
        "Years before 1980",
        "Years after 2026",
        "Negative odometer",
        "Odometer above 500000"
    ],
    "Count": [
        (df["price"] <= 0).sum(),
        (df["year"] < 1980).sum(),
        (df["year"] > 2026).sum(),
        (df["odometer"] < 0).sum(),
        (df["odometer"] > 500000).sum()
    ]
})

outlier_report.to_csv("reports/outlier_report.csv", index=False)

print("\nOutlier report saved successfully!")

# Full dataset analysis using chunks

total_rows = 0
missing_counts = {}
invalid_counts = {
    "Invalid prices": 0,
    "Invalid years": 0,
    "Negative odometer": 0
}

for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):

    total_rows += len(chunk)

    # Count missing values
    for col, count in chunk.isnull().sum().items():
        missing_counts[col] = missing_counts.get(col, 0) + count

    # Check invalid values
    if "price" in chunk.columns:
        price = pd.to_numeric(chunk["price"], errors="coerce")
        invalid_counts["Invalid prices"] += (price <= 0).sum()

    if "year" in chunk.columns:
        year = pd.to_numeric(chunk["year"], errors="coerce")
        invalid_counts["Invalid years"] += (
            (year < 1980) | (year > 2026)
        ).sum()

    if "odometer" in chunk.columns:
        odometer = pd.to_numeric(chunk["odometer"], errors="coerce")
        invalid_counts["Negative odometer"] += (odometer < 0).sum()

# Save full dataset reports
missing_full = pd.DataFrame({
    "Column": list(missing_counts.keys()),
    "Missing Count": list(missing_counts.values())
})

missing_full["Missing Percentage"] = (
    missing_full["Missing Count"] / total_rows * 100
).round(2)

missing_full.sort_values(
    "Missing Percentage", ascending=False
).to_csv("reports/full_dataset_missing_values.csv", index=False)

with open("reports/full_dataset_overview.txt", "w") as f:
    f.write(f"Total records: {total_rows}\n")
    f.write(f"Total columns: {len(missing_counts)}\n")
    f.write("\nInvalid value counts:\n")

    for name, count in invalid_counts.items():
        f.write(f"{name}: {count}\n")

print("\nFULL DATASET ANALYSIS COMPLETED")
print("Total records:", total_rows)
print("Invalid values:", invalid_counts)
print("Reports saved in reports folder.")
