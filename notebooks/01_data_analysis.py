import pandas as pd
import os

recommended = pd.read_csv(
    "../data/raw/works_recommended.csv"
)

sanctioned = pd.read_csv(
    "../data/raw/works_sanctioned.csv"
)

completed = pd.read_csv(
    "../data/raw/works_completed.csv"
)

expenditure = pd.read_csv(
    "../data/raw/expenditure.csv"
)

allocated = pd.read_csv(
    "../data/raw/allocated_limit.csv"
)

calamity = pd.read_csv(
    "../data/raw/calamity.csv"
)

datasets = {
    "Recommended": recommended,
    "Sanctioned": sanctioned,
    "Completed": completed,
    "Expenditure": expenditure,
    "Allocated": allocated,
    "Calamity": calamity
}

for name, df in datasets.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3))