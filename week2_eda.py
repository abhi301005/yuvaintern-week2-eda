# ===== Cell 1 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# ===== Cell 3 =====
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

bank_zip = "bank.zip"

# Download the public dataset
import urllib.request

urllib.request.urlretrieve(url, bank_zip)

print("Bank Marketing dataset downloaded successfully.")

# ===== Cell 4 =====
import zipfile
import os

# Extract the downloaded ZIP file
with zipfile.ZipFile(bank_zip, "r") as zip_ref:
    zip_ref.extractall("bank_data")

print("Dataset extracted successfully.")

# Check extracted files
print(os.listdir("bank_data"))

# ===== Cell 5 =====
data_path = "bank_data/bank-full.csv"

df = pd.read_csv(data_path, sep=";")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ===== Cell 6 =====
print("\nFirst 5 rows:")
display(df.head())

# ===== Cell 7 =====
print("\nColumn names:")
print(df.columns.tolist())

# ===== Cell 8 =====
print("\nDataset shape:")
print(df.shape)

# ===== Cell 9 =====
df.to_csv("bank_marketing_acquired.csv", index=False)

print("Acquired dataset saved as bank_marketing_acquired.csv")

# ===== Cell 11 =====
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

# Number of rows and columns
print("\nDataset dimensions:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Column names
print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# Data types
print("\nData types:")
print(df.dtypes)

# First 10 rows
print("\nFirst 10 records:")
display(df.head(10))

# Random sample
print("\nRandom sample of 5 records:")
display(df.sample(5, random_state=42))

# ===== Cell 12 =====
# DESCRIPTIVE STATISTICS

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

# Numerical variables
print("\nNumerical summary:")
display(df.describe())

# Categorical variables
print("\nCategorical summary:")
display(df.describe(include="object").T)

# ===== Cell 13 =====
# UNIQUE VALUES IN CATEGORICAL FEATURES

categorical_columns = df.select_dtypes(include="object").columns

print("=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in categorical_columns:
    print(f"\n{column}:")
    print("Number of unique values:", df[column].nunique())
    print(df[column].unique())

# ===== Cell 14 =====
# TARGET VARIABLE ANALYSIS

print("=" * 60)
print("TARGET VARIABLE: SUBSCRIPTION")
print("=" * 60)

print("\nSubscription counts:")
display(df["y"].value_counts())

print("\nSubscription percentages:")
display(
    df["y"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ===== Cell 16 =====
print("=" * 60)
print("DATA QUALITY ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 3.1 Missing Values
# ------------------------------------------------------------

print("\n1. Missing values by column:")
missing_values = df.isnull().sum()
print(missing_values)

print("\nTotal missing values:")
print(missing_values.sum())


# ------------------------------------------------------------
# 3.2 Duplicate Records
# ------------------------------------------------------------

print("\n2. Duplicate records:")
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 3.3 Unknown / Special Values
# ------------------------------------------------------------

print("\n3. Special / unknown values:")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    unknown_count = (df[column] == "unknown").sum()

    if unknown_count > 0:
        percentage = (unknown_count / len(df)) * 100

        print(
            f"{column}: {unknown_count} unknown values "
            f"({percentage:.2f}%)"
        )


# ------------------------------------------------------------
# 3.4 Special Values in pdays and previous
# ------------------------------------------------------------

print("\n4. Special values in pdays:")
print("Number of -1 values:", (df["pdays"] == -1).sum())
print(
    "Percentage:",
    round((df["pdays"] == -1).mean() * 100, 2),
    "%"
)

print("\n5. Special values in previous:")
print("Number of 0 values:", (df["previous"] == 0).sum())
print(
    "Percentage:",
    round((df["previous"] == 0).mean() * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 3.5 Numerical Range Check
# ------------------------------------------------------------

print("\n6. Numerical range check:")

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    print(
        f"{column}: "
        f"min={df[column].min()}, "
        f"max={df[column].max()}"
    )

# ===== Cell 18 =====
print("=" * 60)
print("UNIVARIATE EXPLORATORY DATA ANALYSIS")
print("=" * 60)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 4.1 Target Variable Distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="y"
)

plt.title("Distribution of Term Deposit Subscription")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")

# Add percentages above bars
total = len(df)

for container in ax.containers:
    labels = [
        f"{(value / total) * 100:.1f}%"
        for value in container.datavalues
    ]
    ax.bar_label(container, labels=labels, padding=3)

plt.tight_layout()
plt.show()

# ===== Cell 19 =====
# 4.2 Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ===== Cell 20 =====
# 4.3 Balance Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="balance",
    bins=50,
    kde=True
)

plt.title("Distribution of Customer Account Balance")
plt.xlabel("Balance")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ===== Cell 21 =====
# 4.4 Job Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

job_order = df["job"].value_counts().index

sns.countplot(
    data=df,
    y="job",
    order=job_order
)

plt.title("Distribution of Customers by Job")
plt.xlabel("Number of Customers")
plt.ylabel("Job")

plt.tight_layout()
plt.show()

# ===== Cell 22 =====
# 4.5 Education Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

education_order = df["education"].value_counts().index

sns.countplot(
    data=df,
    x="education",
    order=education_order
)

plt.title("Distribution of Customers by Education")
plt.xlabel("Education Level")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# ===== Cell 23 =====
# 4.6 Campaign Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="campaign",
    bins=30
)

plt.title("Distribution of Number of Contacts During Campaign")
plt.xlabel("Campaign Contacts")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ===== Cell 26 =====
print("=" * 60)
print("BIVARIATE EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 5.1 Subscription Rate by Job
# ------------------------------------------------------------

job_subscription = (
    pd.crosstab(
        df["job"],
        df["y"],
        normalize="index"
    ) * 100
)

job_subscription = job_subscription.sort_values(
    by="yes",
    ascending=False
)

print("\nSubscription rate by job:")
display(job_subscription)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=job_subscription.reset_index(),
    x="yes",
    y="job"
)

plt.title("Subscription Rate by Job")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job")
plt.xlim(0, job_subscription["yes"].max() + 5)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 27 =====
# 5.2 Subscription Rate by Education
# ------------------------------------------------------------

education_subscription = (
    pd.crosstab(
        df["education"],
        df["y"],
        normalize="index"
    ) * 100
)

education_subscription = education_subscription.sort_values(
    by="yes",
    ascending=False
)

print("\nSubscription rate by education:")
display(education_subscription)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=education_subscription.reset_index(),
    x="education",
    y="yes"
)

plt.title("Subscription Rate by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ===== Cell 28 =====
# 5.3 Age vs Subscription
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="y",
    y="age"
)

plt.title("Age Distribution by Subscription Outcome")
plt.xlabel("Subscription")
plt.ylabel("Age")

plt.tight_layout()
plt.show()

# ===== Cell 29 =====
# 5.4 Balance vs Subscription
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="y",
    y="balance"
)

plt.title("Account Balance by Subscription Outcome")
plt.xlabel("Subscription")
plt.ylabel("Account Balance")

plt.tight_layout()
plt.show()

# ===== Cell 30 =====
# 5.5 Contact Method vs Subscription
# ------------------------------------------------------------

contact_subscription = (
    pd.crosstab(
        df["contact"],
        df["y"],
        normalize="index"
    ) * 100
)

print("\nSubscription rate by contact method:")
display(contact_subscription)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=contact_subscription.reset_index(),
    x="contact",
    y="yes"
)

plt.title("Subscription Rate by Contact Method")
plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 31 =====
# 5.6 Campaign Contacts vs Subscription
# ------------------------------------------------------------

campaign_subscription = (
    df.groupby("y")["campaign"]
    .mean()
    .reset_index()
)

print("\nAverage campaign contacts by subscription:")
display(campaign_subscription)

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=campaign_subscription,
    x="y",
    y="campaign"
)

plt.title("Average Number of Campaign Contacts by Subscription")
plt.xlabel("Subscription")
plt.ylabel("Average Campaign Contacts")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 33 =====
print("=" * 60)
print("MULTIVARIATE EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 6.1 Create Age Groups
# ------------------------------------------------------------

df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "66+"
    ]
)

age_subscription = (
    pd.crosstab(
        df["age_group"],
        df["y"],
        normalize="index"
    ) * 100
)

print("\nSubscription rate by age group:")
display(age_subscription)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=age_subscription.reset_index(),
    x="age_group",
    y="yes"
)

plt.title("Subscription Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 34 =====
# 6.2 Job and Education vs Subscription
# ------------------------------------------------------------

job_education = (
    df.groupby(
        ["job", "education"],
        observed=True
    )["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .reset_index(name="subscription_rate")
)

print("\nSubscription rate by job and education:")
display(
    job_education.sort_values(
        "subscription_rate",
        ascending=False
    ).head(15)
)

# Pivot table for heatmap
job_education_pivot = job_education.pivot(
    index="job",
    columns="education",
    values="subscription_rate"
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    job_education_pivot,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Subscription Rate by Job and Education")
plt.xlabel("Education Level")
plt.ylabel("Job")

plt.tight_layout()
plt.show()

# ===== Cell 35 =====
# 6.3 Contact Method and Campaign Contacts
# ------------------------------------------------------------

contact_campaign = (
    df.groupby(["contact", "y"])["campaign"]
    .mean()
    .reset_index()
)

print("\nAverage campaign contacts by contact method and subscription:")
display(contact_campaign)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=contact_campaign,
    x="contact",
    y="campaign",
    hue="y"
)

plt.title("Average Campaign Contacts by Contact Method and Subscription")
plt.xlabel("Contact Method")
plt.ylabel("Average Campaign Contacts")
plt.legend(title="Subscription")

plt.tight_layout()
plt.show()

# ===== Cell 36 =====
# 6.4 Previous Campaign Outcome vs Subscription
# ------------------------------------------------------------

poutcome_subscription = (
    pd.crosstab(
        df["poutcome"],
        df["y"],
        normalize="index"
    ) * 100
)

print("\nSubscription rate by previous campaign outcome:")
display(poutcome_subscription)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=poutcome_subscription.reset_index(),
    x="poutcome",
    y="yes"
)

plt.title("Subscription Rate by Previous Campaign Outcome")
plt.xlabel("Previous Campaign Outcome")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 37 =====
# 6.5 Numerical Feature Relationships
# ------------------------------------------------------------

numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

print("\nCorrelation matrix:")
display(
    df[numeric_features].corr().round(2)
)

# ===== Cell 39 =====
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]


# ===== Cell 40 =====
# 7.1 Correlation Matrix
# ------------------------------------------------------------

correlation_matrix = df[numeric_features].corr()

print("\nCorrelation Matrix:")
display(correlation_matrix.round(2))

# ===== Cell 41 =====
# 7.2 Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.xlabel("Features")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ===== Cell 42 =====
# 7.3 Numerical Features vs Subscription
# ============================================================

# Create binary representation of the target
df["subscription_binary"] = df["y"].map({
    "no": 0,
    "yes": 1
})

target_correlation = (
    df[numeric_features + ["subscription_binary"]]
    .corr()["subscription_binary"]
    .drop("subscription_binary")
    .sort_values(ascending=False)
)

print("Correlation of numerical features with subscription:")
display(target_correlation.round(3))

plt.figure(figsize=(9, 5))

target_correlation.sort_values().plot(kind="barh")

plt.title("Correlation of Numerical Features with Subscription")
plt.xlabel("Correlation with Subscription")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# ===== Cell 43 =====
# 7.4 Strongest Numerical Correlations
# ============================================================

corr_pairs = (
    correlation_matrix
    .where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )
    .stack()
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("Strongest numerical feature relationships:")
display(corr_pairs.head(10).round(3))

# ===== Cell 45 =====
print("=" * 60)
print("DATA TRANSFORMATIONS AND AGGREGATIONS")
print("=" * 60)

# ------------------------------------------------------------
# 8.1 Subscription Rate by Month
# ------------------------------------------------------------

month_order = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]

month_subscription = (
    pd.crosstab(
        df["month"],
        df["y"],
        normalize="index"
    ) * 100
)

month_subscription = month_subscription.reindex(month_order)

print("\nSubscription rate by month:")
display(month_subscription)

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    data=month_subscription.reset_index(),
    x="month",
    y="yes",
    order=month_order
)

plt.title("Subscription Rate by Contact Month")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 46 =====
# 8.2 Campaign Intensity Groups
# ------------------------------------------------------------

df["campaign_group"] = pd.cut(
    df["campaign"],
    bins=[0, 1, 2, 3, 5, 10, np.inf],
    labels=[
        "1",
        "2",
        "3",
        "4-5",
        "6-10",
        "11+"
    ]
)

campaign_subscription = (
    pd.crosstab(
        df["campaign_group"],
        df["y"],
        normalize="index"
    ) * 100
)

print("\nSubscription rate by campaign intensity:")
display(campaign_subscription)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=campaign_subscription.reset_index(),
    x="campaign_group",
    y="yes"
)

plt.title("Subscription Rate by Campaign Contact Intensity")
plt.xlabel("Number of Contacts During Campaign")
plt.ylabel("Subscription Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 47 =====
# ------------------------------------------------------------
# 8.3 Average Call Duration by Subscription
# ------------------------------------------------------------

duration_summary = (
    df.groupby("y")["duration"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\nCall duration summary by subscription:")
display(duration_summary)

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=duration_summary.reset_index(),
    x="y",
    y="mean"
)

plt.title("Average Call Duration by Subscription Outcome")
plt.xlabel("Subscription")
plt.ylabel("Average Call Duration (seconds)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
        padding=3
    )

plt.tight_layout()
plt.show()

# ===== Cell 48 =====
# ------------------------------------------------------------
# 8.4 Previous Campaign Outcome + Age Group
# ------------------------------------------------------------

age_poutcome = (
    df.groupby(
        ["age_group", "poutcome"],
        observed=True
    )["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .reset_index(name="subscription_rate")
)

print("\nSubscription rate by age group and previous campaign outcome:")

display(
    age_poutcome.sort_values(
        "subscription_rate",
        ascending=False
    ).head(20)
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=age_poutcome,
    x="age_group",
    y="subscription_rate",
    hue="poutcome"
)

plt.title(
    "Subscription Rate by Age Group and Previous Campaign Outcome"
)

plt.xlabel("Age Group")
plt.ylabel("Subscription Rate (%)")
plt.legend(title="Previous Outcome")

plt.tight_layout()
plt.show()

# ===== Cell 49 =====
# ------------------------------------------------------------
# 8.5 Log-Scale Balance Visualization
# ------------------------------------------------------------

# Shift balance so that negative values can also be displayed
balance_shift = abs(df["balance"].min()) + 1

df["balance_shifted"] = df["balance"] + balance_shift

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="balance_shifted",
    bins=50,
    kde=True
)

plt.xscale("log")

plt.title("Log-Scale Distribution of Customer Balance")
plt.xlabel("Shifted Account Balance (log scale)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ===== Cell 50 =====
# ------------------------------------------------------------
# 8.6 EDA Summary Table
# ------------------------------------------------------------

summary_table = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Subscription Rate",
        "Average Age",
        "Average Balance",
        "Average Call Duration",
        "Average Campaign Contacts",
        "Customers with Previous Contact"
    ],
    "Value": [
        len(df),
        f"{(df['y'] == 'yes').mean() * 100:.2f}%",
        f"{df['age'].mean():.2f}",
        f"{df['balance'].mean():.2f}",
        f"{df['duration'].mean():.2f} sec",
        f"{df['campaign'].mean():.2f}",
        f"{(df['previous'] > 0).sum()}"
    ]
})

print("\nOverall EDA Summary:")
display(summary_table)

# ===== Cell 52 =====
print("=" * 60)
print("FINAL EDA INSIGHTS")
print("=" * 60)

# ------------------------------------------------------------
# 9.1 Overall subscription rate
# ------------------------------------------------------------

overall_rate = (df["y"] == "yes").mean() * 100

print(f"\nOverall subscription rate: {overall_rate:.2f}%")

# ===== Cell 53 =====
# 9.2 Highest and lowest subscription rate by job
# ------------------------------------------------------------

job_rates = (
    df.groupby("job")["subscription_binary"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nHighest subscription-rate jobs:")
display(job_rates.head(5).round(2))

print("\nLowest subscription-rate jobs:")
display(job_rates.tail(5).round(2))

# ===== Cell 54 =====
# 9.3 Highest and lowest subscription rate by month
# ------------------------------------------------------------

month_rates = (
    df.groupby("month")["subscription_binary"]
    .mean()
    .mul(100)
    .reindex(month_order)
)

print("\nSubscription rate by month:")
display(month_rates.round(2))

print("\nBest-performing months:")
display(month_rates.sort_values(ascending=False).head(5).round(2))

print("\nLowest-performing months:")
display(month_rates.sort_values().head(5).round(2))

# ===== Cell 55 =====
# 9.4 Highest and lowest subscription rate by age group
# ------------------------------------------------------------

age_rates = (
    df.groupby("age_group", observed=True)["subscription_binary"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by age group:")
display(age_rates.round(2))

print("\nHighest-performing age groups:")
display(age_rates.sort_values(ascending=False).head(3).round(2))

# ===== Cell 56 =====
# 9.5 Campaign intensity analysis
# ------------------------------------------------------------

campaign_rates = (
    df.groupby("campaign_group", observed=True)["subscription_binary"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by campaign intensity:")
display(campaign_rates.round(2))

# ===== Cell 57 =====
# 9.6 Previous campaign outcome
# ------------------------------------------------------------

previous_rates = (
    df.groupby("poutcome")["subscription_binary"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nSubscription rate by previous campaign outcome:")
display(previous_rates.round(2))

# ===== Cell 58 =====
# 9.7 Duration comparison
# ------------------------------------------------------------

duration_by_outcome = (
    df.groupby("y")["duration"]
    .agg(["mean", "median"])
    .round(2)
)

print("\nCall duration by subscription outcome:")
display(duration_by_outcome)

# ===== Cell 59 =====
# 9.8 Contact method
# ------------------------------------------------------------

contact_rates = (
    df.groupby("contact")["subscription_binary"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nSubscription rate by contact method:")
display(contact_rates.round(2))


print("\n" + "=" * 60)
print("FINAL INSIGHT EXTRACTION COMPLETED")
print("=" * 60)
