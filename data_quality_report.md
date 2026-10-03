# Data Quality Report
## Kaggle House Prices – Advanced Regression Techniques

### 1. Dataset Overview

The dataset used for this data-quality audit is the Kaggle **House Prices – Advanced Regression Techniques** training dataset.

The raw dataset contains:

- **Rows:** 1,460
- **Columns:** 81
- **Target variable:** `SalePrice`
- **Identifier:** `Id`

The cleaned dataset contains **1,460 rows and 86 columns**. The additional five columns are Boolean outlier flags used for auditability.

The target variable `SalePrice` was preserved and was not used to make data-cleaning or outlier-removal decisions.

---

## 2. Initial Data Inspection

The dataset was loaded using Pandas and inspected for:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Duplicate records
- Duplicate identifiers

The initial dataset contained 81 columns and 1,460 records.

The initial missing-value analysis identified missing values in several columns, with the largest proportions occurring in:

| Column | Missing Count | Missing % |
|---|---:|---:|
| `PoolQC` | 1,453 | 99.52% |
| `MiscFeature` | 1,406 | 96.30% |
| `Alley` | 1,369 | 93.77% |
| `Fence` | 1,179 | 80.75% |
| `MasVnrType` | 872 | 59.73% |
| `FireplaceQu` | 690 | 47.26% |
| `LotFrontage` | 259 | 17.74% |
| `GarageType` | 81 | 5.55% |
| `GarageYrBlt` | 81 | 5.55% |
| `GarageFinish` | 81 | 5.55% |
| `GarageQual` | 81 | 5.55% |
| `GarageCond` | 81 | 5.55% |
| `BsmtExposure` | 38 | 2.60% |
| `BsmtFinType2` | 38 | 2.60% |
| `BsmtQual` | 37 | 2.53% |
| `BsmtCond` | 37 | 2.53% |
| `BsmtFinType1` | 37 | 2.53% |
| `MasVnrArea` | 8 | 0.55% |
| `Electrical` | 1 | 0.07% |

Missing values were not handled using one blanket method. Each important column was investigated based on its meaning.

---

# 3. Missing-Value Cleaning Decisions

## 3.1 LotFrontage

### Problem

`LotFrontage` contained **259 missing values**, representing approximately **17.74%** of the dataset.

`LotFrontage` is a numerical measurement, so replacing missing values with a categorical value such as `"None"` would not be appropriate.

### Decision

Missing `LotFrontage` values were filled using the **median LotFrontage of the property's Neighborhood**.

```python
df["LotFrontage"] = df.groupby(
    "Neighborhood"
)["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
```

### Justification

Neighborhood is relevant to lot characteristics. Using the neighborhood median preserves the local distribution better than using one global value for every property.

The median was selected instead of the mean because it is less affected by unusually large or small lots.

### Result

Remaining missing `LotFrontage` values:

**0**

---

## 3.2 Feature-Absence Categorical Columns

Several categorical columns contain missing values because the corresponding feature does not exist for the property.

The following columns were assigned `"None"` when missing:

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`
- `FireplaceQu`
- `GarageType`
- `GarageFinish`
- `GarageQual`
- `GarageCond`
- `BsmtQual`
- `BsmtCond`
- `BsmtExposure`
- `BsmtFinType1`
- `BsmtFinType2`

```python
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
```

### Justification

For these features, a missing value does not necessarily represent missing information.

For example:

- Missing `PoolQC` can mean the property has no pool.
- Missing garage-quality information can mean there is no garage.
- Missing basement-quality information can mean there is no basement.
- Missing `FireplaceQu` can mean there is no fireplace.

Using `"None"` explicitly represents feature absence instead of incorrectly treating it as an unknown value.

---

## 3.3 Garage Numerical Columns

The following numerical garage columns contained missing values:

- `GarageYrBlt`
- `GarageCars`
- `GarageArea`

They were replaced with `0`.

```python
garage_num = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea"
]

df[garage_num] = df[garage_num].fillna(0)
```

### Justification

The missing garage records correspond to properties without a garage.

For `GarageCars` and `GarageArea`, zero directly represents the absence of garage capacity and garage area.

For `GarageYrBlt`, zero is used as a placeholder representing no garage rather than a real construction year.

---

## 3.4 Masonry Veneer Columns

Two related columns required additional investigation:

- `MasVnrType`
- `MasVnrArea`

Some records had missing masonry veneer information.

When both `MasVnrType` and `MasVnrArea` were missing, they were set to:

- `MasVnrType = "None"`
- `MasVnrArea = 0`

```python
both_missing = (
    df["MasVnrType"].isnull() &
    df["MasVnrArea"].isnull()
)

df.loc[both_missing, "MasVnrType"] = "None"
df.loc[both_missing, "MasVnrArea"] = 0
```

For records where `MasVnrArea` was zero but `MasVnrType` was missing, `MasVnrType` was also assigned `"None"`.

For records where `MasVnrArea` was positive but `MasVnrType` was missing, the type could not be safely inferred. Those records were assigned `"Unknown"`.

```python
no_veneer = (
    df["MasVnrType"].isnull() &
    (df["MasVnrArea"] == 0)
)

df.loc[no_veneer, "MasVnrType"] = "None"

unknown_veneer = (
    df["MasVnrType"].isnull() &
    (df["MasVnrArea"] > 0)
)

df.loc[unknown_veneer, "MasVnrType"] = "Unknown"
```

### Justification

This approach avoids fabricating a masonry type when the available information does not support a reliable inference.

A zero veneer area provides evidence that the property has no veneer, while a positive area without a known type means that the existence of veneer is known but its specific type is uncertain.

---

## 3.5 Electrical

`Electrical` contained one missing value.

The missing record was investigated using the available property information and neighborhood patterns.

The neighborhood-based mode was used to determine the most common electrical system for the missing record's neighborhood.

```python
electrical_mode = (
    df.groupby("Neighborhood")["Electrical"]
      .agg(lambda x: x.mode().iloc[0])
)

df.loc[
    df["Electrical"].isnull(),
    "Electrical"
] = electrical_mode["Timber"]
```

The resulting value was `SBrkr`.

### Justification

Only one value was missing, so dropping the record would unnecessarily remove a valid observation.

The neighborhood mode provided a context-based categorical replacement rather than an arbitrary value.

---

# 4. Duplicate Record Audit

Duplicate records were checked using Pandas.

### Results

- Exact duplicate rows: **0**
- Duplicate `Id` values: **0**
- Potential duplicate records after excluding `Id`: **0**

Therefore, no records were removed because of duplication.

The unique `Id` values also confirm that each property has a unique identifier.

---

# 5. Data-Type Correction

The column `MSSubClass` was initially stored as an integer.

However, `MSSubClass` represents a **dwelling/classification code**, rather than a continuous numerical measurement.

It was therefore converted to a string:

```python
df["MSSubClass"] = df["MSSubClass"].astype(str)
```

### Justification

Treating a classification code as a numerical measurement could incorrectly imply that the difference between two codes has mathematical meaning.

The corrected data type is:

**`object`**

---

# 6. Outlier Detection

Outliers were investigated using the **Interquartile Range (IQR)** method.

The following numerical columns were selected:

1. `GrLivArea`
2. `LotArea`
3. `1stFlrSF`
4. `TotalBsmtSF`
5. `GarageArea`

The target variable `SalePrice` was deliberately excluded from the outlier-cleaning process.

The IQR method uses:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

---

## 6.1 Outlier Summary

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers |
|---|---:|---:|---:|---:|---:|---:|
| `GrLivArea` | 1129.50 | 1776.75 | 647.25 | 158.625 | 2747.625 | 31 |
| `LotArea` | 7553.50 | 11601.50 | 4048.00 | 1481.500 | 17673.500 | 69 |
| `1stFlrSF` | 882.00 | 1391.25 | 509.25 | 118.125 | 2155.125 | 20 |
| `TotalBsmtSF` | 795.75 | 1298.25 | 502.50 | 42.000 | 2052.000 | 61 |
| `GarageArea` | 334.50 | 576.00 | 241.50 | -27.750 | 938.250 | 21 |

---

# 7. Outlier Investigation and Decisions

An IQR result alone was not considered sufficient evidence that a record was erroneous.

The flagged records were examined using related housing characteristics.

## `GrLivArea`

Large above-ground living areas were identified as outliers.

The extreme observations generally showed consistency with other housing characteristics such as:

- Overall quality
- Number of rooms
- Number of bathrooms
- First-floor area
- Second-floor area
- Basement area

Therefore, these observations were treated as potentially genuine large properties rather than automatically being classified as data errors.

---

## `LotArea`

Several properties had unusually large lot areas.

Large lots can represent genuine properties rather than incorrect data.

The extreme observations were compared with other property characteristics and were not automatically removed.

Therefore, the records were retained and flagged.

---

## `1stFlrSF`

Some properties had unusually large first-floor areas.

These values were compared with related area variables such as `GrLivArea` and other house-size measurements.

The values generally showed internal consistency, so they were retained and flagged instead of deleted.

---

## `TotalBsmtSF`

Some properties had unusually large basement areas.

The extreme values were compared with other house-area variables.

The observations generally appeared internally consistent with the overall property size.

Therefore, they were retained and flagged.

---

## `GarageArea`

Some properties had unusually large garage areas.

These were examined in relation to:

- `GarageCars`
- `GrLivArea`
- Overall property size

No sufficient evidence was found to classify all flagged values as data-entry errors.

Therefore, they were retained and flagged.

---

# 8. Outlier Handling Decision

The identified outliers were **not removed**.

Instead, five Boolean audit columns were created:

- `GrLivArea_Outlier`
- `LotArea_Outlier`
- `1stFlrSF_Outlier`
- `TotalBsmtSF_Outlier`
- `GarageArea_Outlier`

The resulting counts were:

| Outlier Flag | Count |
|---|---:|
| `GrLivArea_Outlier` | 31 |
| `LotArea_Outlier` | 69 |
| `1stFlrSF_Outlier` | 20 |
| `TotalBsmtSF_Outlier` | 61 |
| `GarageArea_Outlier` | 21 |

### Reason for retaining outliers

The IQR method identifies statistical extremes, but statistical extremeness does not automatically mean that a value is incorrect.

The housing records were therefore preserved unless there was sufficient evidence of a data-entry or measurement error.

The Boolean flags maintain traceability and allow later modelling steps to decide how to treat the observations.

---

# 9. Target Variable Protection

`SalePrice` was treated as the target variable.

The final checks showed:

- Missing `SalePrice` values: **0**
- Data type: **int64**
- Number of records: **1,460**

`SalePrice` was not used to determine missing-value replacements or identify outliers.

This avoids allowing target information to influence the data-cleaning process.

---

# 10. Final Data Quality Check

After cleaning:

| Quality Check | Result |
|---|---:|
| Rows | 1,460 |
| Columns | 86 |
| Remaining missing values | **0** |
| Duplicate rows | **0** |
| Duplicate IDs | **0** |
| `MSSubClass` data type | `object` |
| `SalePrice` missing values | **0** |
| Outlier flags | **5** |

The cleaned dataset was saved as:

```text
train_cleaned.csv
```

The saved file was then reloaded using:

```python
pd.read_csv(
    "train_cleaned.csv",
    keep_default_na=False
)
```

This was necessary because the value `"None"` was intentionally used to represent feature absence. Pandas can otherwise interpret `"None"` as a missing value when reading the CSV.

The saved-file verification confirmed:

- Shape: **1,460 × 86**
- Missing values: **0**
- Duplicate rows: **0**
- Duplicate IDs: **0**
- `SalePrice` missing values: **0**
- Outlier flags retained correctly

---

# 11. Final Conclusion

The Kaggle House Prices training dataset was audited and cleaned using a column-specific data-quality approach.

The main decisions were:

- Context-based median imputation for `LotFrontage`
- `"None"` for categorical features that are genuinely absent
- Zero for numerical garage fields when no garage exists
- Context-based handling of masonry veneer information
- Neighborhood-based mode for the single missing `Electrical` value
- Conversion of `MSSubClass` from integer to categorical/string representation
- Duplicate checks with no duplicate records found
- IQR-based outlier detection across five numerical variables
- Retention and flagging of outliers rather than automatic deletion
- Preservation of `SalePrice` without using it for cleaning decisions

The resulting `train_cleaned.csv` contains all 1,460 original observations, with no remaining missing values and with five additional outlier-audit columns.
