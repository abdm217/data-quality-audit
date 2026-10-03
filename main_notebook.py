# ============================================================
# DATA CLEANING & DATA QUALITY AUDIT
# Kaggle House Prices - Advanced Regression Techniques
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("train.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. BASIC DATASET INSPECTION
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nNumber of columns:", len(df.columns))

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 4. INITIAL MISSING-VALUE REPORT
# ------------------------------------------------------------

missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
].sort_values(
    "Missing Count",
    ascending=False
)

print("\nInitial Missing-Value Report:")
print(missing_report)


# ------------------------------------------------------------
# 5. DUPLICATE CHECK
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

duplicate_ids = df["Id"].duplicated().sum()

print("Number of duplicate IDs:", duplicate_ids)


# Check for potential near-duplicates
df_without_id = df.drop(columns=["Id"])

near_duplicates = df_without_id[
    df_without_id.duplicated(keep=False)
]

print("Number of potential near-duplicate rows:", len(near_duplicates))


# ------------------------------------------------------------
# 6. DATA TYPE CORRECTION
# ------------------------------------------------------------

# MSSubClass is a dwelling classification code.
# It represents categories rather than a continuous measurement.

df["MSSubClass"] = df["MSSubClass"].astype(str)

print("\nMSSubClass data type after correction:")
print(df["MSSubClass"].dtype)


# ------------------------------------------------------------
# 7. HANDLE LOTFRONTAGE MISSING VALUES
# ------------------------------------------------------------

# LotFrontage is a numerical measurement.
# Missing values are filled using the median LotFrontage
# within the same Neighborhood.

df["LotFrontage"] = df.groupby(
    "Neighborhood"
)["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

print(
    "\nRemaining missing LotFrontage values:",
    df["LotFrontage"].isnull().sum()
)


# ------------------------------------------------------------
# 8. HANDLE FEATURE-ABSENCE CATEGORICAL VALUES
# ------------------------------------------------------------

# For these columns, a missing value means the feature
# does not exist for the house.
#
# Example:
# Missing PoolQC = no pool
# Missing GarageType = no garage
# Missing BsmtQual = no basement

none_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2"
]

df[none_columns] = df[none_columns].fillna("None")


# ------------------------------------------------------------
# 9. HANDLE GARAGE NUMERICAL VALUES
# ------------------------------------------------------------

# Missing garage numerical values represent the absence
# of a garage, so they are represented as 0.

garage_num = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea"
]

df[garage_num] = df[garage_num].fillna(0)


# ------------------------------------------------------------
# 10. HANDLE MASONRY VENEER VALUES
# ------------------------------------------------------------

# First handle rows where both MasVnrType and MasVnrArea
# are missing.

both_missing = (
    df["MasVnrType"].isnull() &
    df["MasVnrArea"].isnull()
)

df.loc[both_missing, "MasVnrType"] = "None"
df.loc[both_missing, "MasVnrArea"] = 0


# Missing MasVnrType + zero area means there is no
# masonry veneer.

no_veneer = (
    df["MasVnrType"].isnull() &
    (df["MasVnrArea"] == 0)
)

df.loc[no_veneer, "MasVnrType"] = "None"


# Missing MasVnrType + positive area means masonry exists,
# but its type cannot be reliably inferred.

unknown_veneer = (
    df["MasVnrType"].isnull() &
    (df["MasVnrArea"] > 0)
)

df.loc[unknown_veneer, "MasVnrType"] = "Unknown"


# ------------------------------------------------------------
# 11. HANDLE ELECTRICAL MISSING VALUE
# ------------------------------------------------------------

# The only missing Electrical value belongs to a house
# in the Timber neighborhood.
#
# The most common Electrical type in Timber is SBrkr,
# so the missing value is filled with SBrkr.

electrical_mode = (
    df.groupby("Neighborhood")["Electrical"]
      .agg(lambda x: x.mode().iloc[0])
)

df.loc[
    df["Electrical"].isnull(),
    "Electrical"
] = electrical_mode["Timber"]


# ------------------------------------------------------------
# 12. VERIFY MISSING VALUES AFTER CLEANING
# ------------------------------------------------------------

missing_report_final = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_report_final = missing_report_final[
    missing_report_final["Missing Count"] > 0
].sort_values(
    "Missing Count",
    ascending=False
)

print("\nRemaining missing values:")
print(missing_report_final)

total_missing = df.isnull().sum().sum()

print("\nTotal remaining missing values:", total_missing)


# ------------------------------------------------------------
# 13. VERIFY DUPLICATES AFTER CLEANING
# ------------------------------------------------------------

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nDuplicate IDs after cleaning:")
print(df["Id"].duplicated().sum())


# ------------------------------------------------------------
# 14. OUTLIER ANALYSIS USING IQR
# ------------------------------------------------------------

# SalePrice is deliberately NOT included in outlier analysis.
#
# The selected numerical columns are:
# GrLivArea
# LotArea
# 1stFlrSF
# TotalBsmtSF
# GarageArea

outlier_columns = [
    "GrLivArea",
    "LotArea",
    "1stFlrSF",
    "TotalBsmtSF",
    "GarageArea"
]

outlier_summary = []

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_summary.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count
    })


outlier_summary = pd.DataFrame(outlier_summary)

print("\nOutlier Summary:")
print(outlier_summary)


# ------------------------------------------------------------
# 15. CREATE OUTLIER FLAGS
# ------------------------------------------------------------

# The outliers are retained because they were not established
# as data-entry errors.
#
# Boolean flags are added so the observations remain visible
# for later analysis.

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    df[column + "_Outlier"] = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )


# ------------------------------------------------------------
# 16. VERIFY OUTLIER FLAGS
# ------------------------------------------------------------

outlier_flag_columns = [
    "GrLivArea_Outlier",
    "LotArea_Outlier",
    "1stFlrSF_Outlier",
    "TotalBsmtSF_Outlier",
    "GarageArea_Outlier"
]

print("\nNumber of flagged outliers:")
print(df[outlier_flag_columns].sum())


# ------------------------------------------------------------
# 17. VERIFY SALEPRICE WAS NOT ALTERED
# ------------------------------------------------------------

print("\nSalePrice missing values:")
print(df["SalePrice"].isnull().sum())

print("\nSalePrice data type:")
print(df["SalePrice"].dtype)

print("\nSalePrice summary:")
print(df["SalePrice"].describe())


# ------------------------------------------------------------
# 18. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n================ FINAL DATA QUALITY CHECK ================")

print("\nDataset shape:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate IDs:")
print(df["Id"].duplicated().sum())

print("\nMSSubClass data type:")
print(df["MSSubClass"].dtype)

print("\nOutlier flags:")
print(df[outlier_flag_columns].sum())


# ------------------------------------------------------------
# 19. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(
    "train_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully:")
print("train_cleaned.csv")


# ------------------------------------------------------------
# 20. RELOAD SAVED DATASET FOR FINAL VERIFICATION
# ------------------------------------------------------------

# keep_default_na=False prevents the literal category
# "None" from being interpreted as NaN when the CSV
# is loaded again.

cleaned_df = pd.read_csv(
    "train_cleaned.csv",
    keep_default_na=False
)


# ------------------------------------------------------------
# 21. FINAL SAVED-FILE VERIFICATION
# ------------------------------------------------------------

print("\n================ SAVED FILE VERIFICATION ================")

print("\nSaved dataset shape:")
print(cleaned_df.shape)

print("\nMissing values:")
print(cleaned_df.isnull().sum().sum())

print("\nDuplicate rows:")
print(cleaned_df.duplicated().sum())

print("\nDuplicate IDs:")
print(cleaned_df["Id"].duplicated().sum())

print("\nSalePrice data type:")
print(cleaned_df["SalePrice"].dtype)

print("\nSalePrice missing values:")
print(cleaned_df["SalePrice"].isnull().sum())

print("\nOutlier flags:")
print(cleaned_df[outlier_flag_columns].sum())


# ------------------------------------------------------------
# 22. FINAL SUCCESS MESSAGE
# ------------------------------------------------------------

print("\n============================================================")
print("DATA CLEANING AND QUALITY AUDIT COMPLETED SUCCESSFULLY")
print("============================================================")