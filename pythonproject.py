import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"D:\Electric_Vehicle_Population_Data.csv")
sns.set(style="whitegrid")
df_clean = df.copy()
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, y='Electric Vehicle Type', order=df_clean['Electric Vehicle Type'].value_counts().index)
plt.title("Distribution of Electric Vehicle Types")
plt.xlabel("Count")
plt.ylabel("Vehicle Type")
plt.tight_layout()
plt.show()



print("Columns in the dataset:")
print(df.columns)
numeric_cols = ['Model Year', 'Electric Range', 'Base MSRP']
summary_stats = df[numeric_cols].describe()
print("\nSummary Statistics:")
print(summary_stats)
print("\nMissing Values:")
print(df[numeric_cols].isnull().sum())


df_filtered = df[['Electric Range', 'Base MSRP']].dropna()
correlation = df_filtered.corr()
print("Correlation Matrix:")
print(correlation)
plt.figure(figsize=(8, 6))
sns.regplot(x='Electric Range', y='Base MSRP', data=df_filtered, scatter_kws={'alpha':0.5})
plt.title('Electric Range vs Base MSRP')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Base MSRP ($)')
plt.tight_layout()
plt.show()



df_range = df[['Electric Range']].dropna()
plt.figure(figsize=(8, 4))
sns.boxplot(x='Electric Range', data=df_range)
plt.title('Boxplot of Electric Range')
plt.xlabel('Electric Range (miles)')
plt.tight_layout()
plt.show()
Q1 = df_range['Electric Range'].quantile(0.25)
Q3 = df_range['Electric Range'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_range[(df_range['Electric Range'] < lower_bound) | (df_range['Electric Range'] > upper_bound)]
print(f"Number of outliers in Electric Range: {len(outliers)}")
print(outliers.head())



city_counts = df['City'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=city_counts.values, y=city_counts.index)
plt.title("Top 10 Cities with Most Electric Vehicles")
plt.xlabel("Number of EVs")
plt.ylabel("City")
plt.tight_layout()
plt.show()
county_counts = df['County'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=county_counts.values, y=county_counts.index)
plt.title("Top 10 Counties with Most Electric Vehicles")
plt.xlabel("Number of EVs")
plt.ylabel("County")
plt.tight_layout()
plt.show()

