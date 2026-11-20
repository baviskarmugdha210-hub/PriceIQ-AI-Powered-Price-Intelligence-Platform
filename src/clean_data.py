
import pandas as pd
import os

input_file = "data/raw/vehicles.csv"
output_file = "data/processed/vehicles_clean.csv"

os.makedirs("data/processed", exist_ok=True)

# Remove an older output file before starting
if os.path.exists(output_file):
    os.remove(output_file)

total_rows = 0
clean_rows = 0

for chunk in pd.read_csv(input_file, chunksize=50000, low_memory=False):

    total_rows += len(chunk)

    # Remove duplicate records within each chunk
    chunk = chunk.drop_duplicates()

    # Convert important columns to numeric
    for col in ["price", "year", "odometer"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Remove records without a valid price or year
    chunk = chunk.dropna(subset=["price", "year"])

    # Remove invalid values
    chunk = chunk[
        (chunk["price"] > 0) &
        (chunk["year"] >= 1900) &
        (chunk["year"] <= 2026)
    ]

    # Remove negative odometer readings
    if "odometer" in chunk.columns:
        chunk = chunk[
            chunk["odometer"].isna() |
            (chunk["odometer"] >= 0)
        ]

    clean_rows += len(chunk)

    # Append cleaned data to output CSV
    chunk.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not os.path.exists(output_file)
    )

print("\nDATA CLEANING COMPLETED")
print("Original records processed:", total_rows)
print("Clean records retained:", clean_rows)
print("Records removed:", total_rows - clean_rows)
print("Cleaned file:", output_file)
