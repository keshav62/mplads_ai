import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os
import re



# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to raw data folder
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


# Load datasets

recommended = pd.read_csv(
    os.path.join(DATA_DIR, "works_recommended.csv")
)

sanctioned = pd.read_csv(
    os.path.join(DATA_DIR, "works_sanctioned.csv")
)

completed = pd.read_csv(
    os.path.join(DATA_DIR, "works_completed.csv")
)

expenditure = pd.read_csv(
    os.path.join(DATA_DIR, "expenditure.csv")
)

allocated = pd.read_csv(
    os.path.join(DATA_DIR, "allocated_limit.csv")
)

calamity = pd.read_csv(
    os.path.join(DATA_DIR, "calamity.csv")
)


print("All datasets loaded successfully!")

print("\nRecommended:", recommended.shape)
print("Sanctioned:", sanctioned.shape)
print("Completed:", completed.shape)
print("Expenditure:", expenditure.shape)
print("Allocated:", allocated.shape)
print("Calamity:", calamity.shape)

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )

    return df

recommended = clean_columns(recommended)
sanctioned = clean_columns(sanctioned)
completed = clean_columns(completed)
expenditure = clean_columns(expenditure)
allocated = clean_columns(allocated)
calamity = clean_columns(calamity)

print(recommended.columns)
print(sanctioned.columns)
print(completed.columns)
print(expenditure.columns)


for name, df in {
    "recommended": recommended,
    "sanctioned": sanctioned,
    "completed": completed,
    "expenditure": expenditure
}.items():

    print("\n", name.upper())

    for col in df.columns:
        if "work" in col or "id" in col:
            print(col)


print("\nRECOMMENDED WORK EXAMPLES:")
print(recommended["work"].head(10).to_list())

print("\nSANCTIONED WORK EXAMPLES:")
print(sanctioned["work"].head(10).to_list())

print("\nCOMPLETED WORK EXAMPLES:")
print(completed["work"].head(10).to_list())

print("\nEXPENDITURE WORK EXAMPLES:")
print(expenditure["work"].head(10).to_list())

print("\nEXPENDITURE WORK ID EXAMPLES:")
print(expenditure["work_id"].head(10).to_list())



def extract_work_id(text):
    """
    Extract and clean Work ID from the work column.
    
    Example:
    WS/	 MP620/2024-2025/133166-Construction...
    
    Returns:
    WS/MP620/2024-2025/133166
    """

    if pd.isna(text):
        return None

    # Convert to string
    text = str(text)

    # Remove tabs and extra spaces
    text = re.sub(r"\s+", "", text)

    # Pattern for MPLADS Work ID
    pattern = r"WS/MP\d+/\d{4}-\d{4}/\d+"

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group().upper()

    return None

recommended["work_id"] = recommended["work"].apply(extract_work_id)

sanctioned["work_id"] = sanctioned["work"].apply(extract_work_id)

completed["work_id"] = completed["work"].apply(extract_work_id)


def clean_work_id(work_id):

    if pd.isna(work_id):
        return None

    work_id = str(work_id)

    # Remove all whitespace
    work_id = re.sub(r"\s+", "", work_id)

    # Convert to uppercase
    return work_id.upper()


expenditure["work_id"] = expenditure["work_id"].apply(clean_work_id)

print("\n" + "=" * 80)
print("WORK ID EXTRACTION CHECK")
print("=" * 80)

print("\nRecommended:")
print(recommended[["work", "work_id"]].head())

print("\nSanctioned:")
print(sanctioned[["work", "work_id"]].head())

print("\nCompleted:")
print(completed[["work", "work_id"]].head())

print("\nExpenditure:")
print(expenditure[["work_id"]].head())



print("\nRECOMMENDED INVALID ROWS:")
print(
    recommended[
        recommended["work_id"].isna()
    ].to_string()
)

print("\nSANCTIONED INVALID ROWS:")
print(
    sanctioned[
        sanctioned["work_id"].isna()
    ].to_string()
)

print("\nCOMPLETED INVALID ROWS:")
print(
    completed[
        completed["work_id"].isna()
    ].to_string()
)


recommended = recommended.dropna(subset=["work_id"])

sanctioned = sanctioned.dropna(subset=["work_id"])

completed = completed.dropna(subset=["work_id"])

expenditure = expenditure.dropna(subset=["work_id"])


print("\n" + "=" * 80)
print("UNIQUE WORK ID COUNTS")
print("=" * 80)

print(
    "Recommended:",
    recommended["work_id"].nunique()
)

print(
    "Sanctioned:",
    sanctioned["work_id"].nunique()
)

print(
    "Completed:",
    completed["work_id"].nunique()
)

print(
    "Expenditure:",
    expenditure["work_id"].nunique()
)



recommended_ids = set(recommended["work_id"])

sanctioned_ids = set(sanctioned["work_id"])

completed_ids = set(completed["work_id"])

expenditure_ids = set(expenditure["work_id"])


rec_san_matches = recommended_ids & sanctioned_ids

print(
    "\nRecommended → Sanctioned matches:",
    len(rec_san_matches)
)


san_comp_matches = sanctioned_ids & completed_ids

print(
    "Sanctioned → Completed matches:",
    len(san_comp_matches)
)


rec_comp_matches = recommended_ids & completed_ids

print(
    "Recommended → Completed matches:",
    len(rec_comp_matches)
)


comp_exp_matches = completed_ids & expenditure_ids

print(
    "Completed → Expenditure matches:",
    len(comp_exp_matches)
)

all_matches = (
    recommended_ids
    & sanctioned_ids
    & completed_ids
    & expenditure_ids
)

print(
    "\nPresent in ALL FOUR datasets:",
    len(all_matches)
)

recommended = recommended.dropna(subset=["work_id"])
sanctioned = sanctioned.dropna(subset=["work_id"])
completed = completed.dropna(subset=["work_id"])
expenditure = expenditure.dropna(subset=["work_id"])


print("\n" + "=" * 80)
print("DUPLICATE WORK ID ANALYSIS")
print("=" * 80)

print(
    "Recommended duplicate work IDs:",
    recommended["work_id"].duplicated().sum()
)

print(
    "Sanctioned duplicate work IDs:",
    sanctioned["work_id"].duplicated().sum()
)

print(
    "Completed duplicate work IDs:",
    completed["work_id"].duplicated().sum()
)

print(
    "Expenditure duplicate work IDs:",
    expenditure["work_id"].duplicated().sum()
)

recommended_duplicates = recommended[
    recommended["work_id"].duplicated(keep=False)
]

print("\nRECOMMENDED DUPLICATE EXAMPLES:")

print(
    recommended_duplicates[
        [
            "work_id",
            "state",
            "constituency",
            "recommended_amount_____",
            "recommended_date"
        ]
    ].head(20)
)

print(expenditure["fund_disbursed_amount___"].dtype)

print(
    expenditure["fund_disbursed_amount___"].head(10)
)

def clean_amount(value):

    if pd.isna(value):
        return None

    value = str(value)

    # Remove commas
    value = value.replace(",", "")

    # Remove rupee symbol
    value = value.replace("₹", "")

    # Remove spaces
    value = value.strip()

    return pd.to_numeric(value, errors="coerce")


expenditure["fund_disbursed_amount___"] = (
    expenditure["fund_disbursed_amount___"]
    .apply(clean_amount)
)


expenditure_summary = (
    expenditure
    .groupby("work_id")
    .agg(
        total_expenditure=(
            "fund_disbursed_amount___",
            "sum"
        ),

        payment_count=(
            "work_id",
            "count"
        ),

        unique_vendors=(
            "vendor_name",
            "nunique"
        )
    )
    .reset_index()
)


expenditure["fund_disbursed_amount___"] = pd.to_numeric(
    expenditure["fund_disbursed_amount___"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip(),
    errors="coerce"
)

print("\n" + "=" * 80)
print("EXPENDITURE AMOUNT CHECK")
print("=" * 80)

print(expenditure["fund_disbursed_amount___"].dtype)

print(
    expenditure["fund_disbursed_amount___"].head(10)
)

print(
    "\nMissing expenditure amounts:",
    expenditure["fund_disbursed_amount___"].isna().sum()
)

print("\n" + "=" * 80)
print("EXPENDITURE SUMMARY")
print("=" * 80)

print("Shape:", expenditure_summary.shape)

print("\nFirst 10 records:")

print(expenditure_summary.head(10))

print(
    "\nUnique work IDs:",
    expenditure_summary["work_id"].nunique()
)

print("\n" + "=" * 80)
print("INVALID EXPENDITURE WORK IDs")
print("=" * 80)

invalid_expenditure = expenditure[
    expenditure["work_id"].isna()
    | (expenditure["work_id"].str.strip() == "")
]

print(invalid_expenditure.to_string())


# Remove missing work IDs
expenditure = expenditure.dropna(subset=["work_id"])

# Remove empty work IDs
expenditure = expenditure[
    expenditure["work_id"].str.strip() != ""
]

expenditure_summary = (
    expenditure
    .groupby("work_id")
    .agg(
        total_expenditure=(
            "fund_disbursed_amount___",
            "sum"
        ),

        payment_count=(
            "work_id",
            "count"
        ),

        unique_vendors=(
            "vendor_name",
            "nunique"
        )
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("CLEANED EXPENDITURE SUMMARY")
print("=" * 80)

print("Shape:", expenditure_summary.shape)

print(expenditure_summary.head(10))

print("\nEmpty Work IDs:",
      (expenditure_summary["work_id"] == "").sum()
)


# Keep only records with valid work_id

recommended = recommended.dropna(subset=["work_id"])
sanctioned = sanctioned.dropna(subset=["work_id"])
completed = completed.dropna(subset=["work_id"])

# Remove empty strings
recommended = recommended[
    recommended["work_id"].str.strip() != ""
]

sanctioned = sanctioned[
    sanctioned["work_id"].str.strip() != ""
]

completed = completed[
    completed["work_id"].str.strip() != ""
]
def clean_amount(value):

    if pd.isna(value):
        return None

    value = str(value)

    # Remove commas
    value = value.replace(",", "")

    # Remove rupee symbol
    value = value.replace("₹", "")

    # Remove spaces
    value = value.strip()

    return pd.to_numeric(value, errors="coerce")


recommended["recommended_amount_____"] = (
    recommended["recommended_amount_____"]
    .apply(clean_amount)
)

sanctioned["sanction_amount___"] = (
    sanctioned["sanction_amount___"]
    .apply(clean_amount)
)

completed["amount_disbursed___"] = (
    completed["amount_disbursed___"]
    .apply(clean_amount)
)


recommended_clean = recommended[
    [
        "work_id",
        "state",
        "ida",
        "honble_members_of_parliament",
        "constituency",
        "work_category",
        "work_description",
        "recommended_date",
        "recommended_amount_____"
    ]
].copy()


recommended_clean.rename(
    columns={
        "recommended_amount_____": "recommended_amount"
    },
    inplace=True
)

sanctioned_clean = sanctioned[
    [
        "work_id",
        "sanction_date",
        "sanction_amount___",
        "work_status"
    ]
].copy()

sanctioned_clean.rename(
    columns={
        "sanction_amount___": "sanction_amount"
    },
    inplace=True
)

completed_clean = completed[
    [
        "work_id",
        "completion_date",
        "amount_disbursed___"
    ]
].copy()

completed_clean.rename(
    columns={
        "amount_disbursed___": "completed_amount_disbursed"
    },
    inplace=True
)

master_df = recommended_clean.merge(
    sanctioned_clean,
    on="work_id",
    how="left"
)

master_df = master_df.merge(
    completed_clean,
    on="work_id",
    how="left"
)

master_df = master_df.merge(
    expenditure_summary,
    on="work_id",
    how="left"
)

print("\n" + "=" * 80)
print("MASTER DATASET")
print("=" * 80)

print("Shape:", master_df.shape)

print("\nColumns:")

print(master_df.columns.tolist())

print("\nFirst 5 rows:")

print(master_df.head())

print("\n" + "=" * 80)
print("LIFECYCLE DATA AVAILABILITY")
print("=" * 80)

print(
    "Total recommended works:",
    len(master_df)
)

print(
    "Has sanction:",
    master_df["sanction_date"].notna().sum()
)

print(
    "Has completion:",
    master_df["completion_date"].notna().sum()
)

print(
    "Has expenditure:",
    master_df["total_expenditure"].notna().sum()
)


print("\n" + "=" * 80)
print("DATE CLEANING")
print("=" * 80)

date_columns = [
    "recommended_date",
    "sanction_date",
    "completion_date"
]

for column in date_columns:
    master_df[column] = pd.to_datetime(
        master_df[column],
        errors="coerce",
        dayfirst=True
    )

print(master_df[date_columns].dtypes)

print("\n" + "=" * 80)
print("CREATING LIFECYCLE FEATURES")
print("=" * 80)

master_df["has_sanction"] = (
    master_df["sanction_date"].notna()
).astype(int)

master_df["has_completion"] = (
    master_df["completion_date"].notna()
).astype(int)

master_df["has_expenditure"] = (
    master_df["total_expenditure"].notna()
).astype(int)

master_df["sanction_delay_days"] = (
    master_df["sanction_date"]
    - master_df["recommended_date"]
).dt.days

master_df["completion_duration_days"] = (
    master_df["completion_date"]
    - master_df["sanction_date"]
).dt.days

master_df["sanction_difference"] = (
    master_df["sanction_amount"]
    - master_df["recommended_amount"]
)

master_df["expenditure_ratio"] = (
    master_df["total_expenditure"]
    / master_df["sanction_amount"]
)

master_df["sanction_ratio"] = (
    master_df["sanction_amount"]
    / master_df["recommended_amount"]
)

master_df["expenditure_difference"] = (
    master_df["total_expenditure"]
    - master_df["sanction_amount"]
)

master_df["recommended_amount"] = master_df[
    "recommended_amount"
].replace(0, np.nan)


master_df["recommended_amount"] = master_df[
    "recommended_amount"
].replace(0, np.nan)

master_df["sanction_amount"] = master_df[
    "sanction_amount"
].replace(0, np.nan)

master_df["completion_amount_ratio"] = (
    master_df["completed_amount_disbursed"]
    / master_df["sanction_amount"]
)

master_df["completion_amount_difference"] = (
    master_df["completed_amount_disbursed"]
    - master_df["sanction_amount"]
)

master_df["payments_per_vendor"] = (
    master_df["payment_count"]
    / master_df["unique_vendors"]
)

master_df["unique_vendors"] = master_df[
    "unique_vendors"
].replace(0, np.nan)

master_df["financial_year"] = (
    master_df["work_id"]
    .str.extract(r"(20\d{2}-20\d{2})")
)

print(
    master_df["financial_year"]
    .value_counts()
)


master_df["recommended_year"] = (
    master_df["recommended_date"].dt.year
)

master_df["recommended_month"] = (
    master_df["recommended_date"].dt.month
)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)

feature_columns = [
    "has_sanction",
    "has_completion",
    "has_expenditure",
    "sanction_delay_days",
    "completion_duration_days",
    "sanction_difference",
    "sanction_ratio",
    "expenditure_difference",
    "expenditure_ratio",
    "completion_amount_ratio",
    "completion_amount_difference",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor",
    "financial_year"
]

print(
    master_df[feature_columns].head(10)
)

print("\nFeature Dataset Shape:")

print(master_df.shape)

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

check_columns = [
    "sanction_delay_days",
    "completion_duration_days",
    "sanction_difference",
    "sanction_ratio",
    "expenditure_difference",
    "expenditure_ratio",
    "completion_amount_ratio",
    "completion_amount_difference",
    "payments_per_vendor"
]

print(
    master_df[check_columns].describe()
)

print("\nNegative Sanction Delays:")

print(
    (master_df["sanction_delay_days"] < 0).sum()
)

print("\nNegative Completion Durations:")

print(
    (master_df["completion_duration_days"] < 0).sum()
)


print("\n" + "=" * 80)
print("FEATURE VARIATION CHECK")
print("=" * 80)

check_features = [
    "recommended_amount",
    "sanction_amount",
    "sanction_difference",
    "sanction_ratio",
    "completed_amount_disbursed",
    "completion_amount_ratio",
    "total_expenditure",
    "expenditure_ratio",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor"
]

for column in check_features:

    print(f"\n{column}")

    print("Unique values:", master_df[column].nunique())

    print(
        master_df[column]
        .dropna()
        .value_counts()
        .head(10)
    )

print("\nSANCTION RATIO SUMMARY")

print(
    master_df["sanction_ratio"]
    .describe()
)

print("\nEXPENDITURE RATIO SUMMARY")

print(
    master_df["expenditure_ratio"]
    .describe()
)

# Create dataset for financial anomaly detection

financial_df = master_df[
    master_df["total_expenditure"].notna()
].copy()

# Create a feature indicating whether sanction data exists

financial_df["has_sanction"] = (
    financial_df["sanction_amount"].notna()
).astype(int)

# Select useful financial features

financial_features = [
    "recommended_amount",
    "sanction_amount",
    "total_expenditure",
    "expenditure_ratio",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor", 
    "has_sanction"
]


# Create training dataset

X_financial = financial_df[financial_features].copy()


print("Training dataset shape:", X_financial.shape)

print("\nMissing values:")
print(X_financial.isnull().sum())

X_financial = X_financial.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values using median

X_financial = X_financial.fillna(
    X_financial.median()
)


# Final check

print("\nMissing values after cleaning:")

print(X_financial.isnull().sum())





# ==========================================
# CREATE SCALER
# ==========================================

scaler = RobustScaler()


# ==========================================
# SCALE THE FINANCIAL FEATURES
# ==========================================

X_financial_scaled = scaler.fit_transform(
    X_financial
)


print(
    "Scaled dataset shape:",
    X_financial_scaled.shape
)


# ==========================================
# CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "models/robust_scaler.pkl"
)


print(
    "Scaler saved successfully!"
)


# ==========================================
# SAVE MASTER DATASET
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


master_df.to_csv(
    "data/processed/master_dataset.csv",
    index=False
)


print(
    "Master dataset saved successfully!"
)