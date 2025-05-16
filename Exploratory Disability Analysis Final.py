# Import necessary libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ===============================================================
# DISABILITY DATA ANALYSIS WITH TRANSPARENT METHODOLOGY
# ===============================================================
# This script analyzes disability-related data from the CDC, applying critical
# principles that question what I measure, how I measure it, and whose stories
# might be missing from my analysis.
#
# MY METHODOLOGICAL CHOICES AND LIMITATIONS (Nullius in verba - take nobody's word for it):
# - I use CDC's self-reported survey data, which has inherent sampling limitations
# - My analysis is constrained to pre-defined disability categories created by the CDC
# - I acknowledge that disability exists on a spectrum and categorical data may oversimplify
# - The 5000-row limit may introduce sampling bias, particularly for smaller states
# - I prioritize transparency over neat conclusions where appropriate
# ===============================================================

# Step 1: Fetch Data - Being explicit about source and limitations
print(f"Data retrieval started at {datetime.now().strftime('%H:%M:%S')}")
url = "https://data.cdc.gov/resource/s2qv-b27b.json?$limit=5000"
response = requests.get(url)
data = response.json()
print(f"Retrieved {len(data)} records out of potentially many more - this limitation may affect my conclusions")
print(f"Data source: CDC API - I acknowledge this data is collected through self-reporting")

# Step 2: Data Normalization with transparent transformations
df = pd.json_normalize(data)
original_columns = df.columns.tolist()
print(f"Original dataset contains {len(original_columns)} variables, most of which I will not analyze")

# Step 3: Select Relevant Columns - Being explicit about what I choose to focus on and why
cols = ['year', 'locationabbr', 'locationdesc', 'response', 'data_value']
df = df[cols]
df.columns = ['Year', 'StateAbbr', 'State', 'DisabilityType', 'Rate']
print(f"I've reduced the data to {len(df.columns)} variables, which means I've excluded many aspects")
print("This choice reflects my specific focus but inevitably leaves out other important dimensions")

# Step 4: Data Cleaning - Document transformations and their potential impacts
missing_before = df.isna().sum().sum()
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df.dropna(subset=['Year', 'Rate'], inplace=True)
missing_after = df.isna().sum().sum()
print(f"Removed {len(df) - df.dropna().shape[0]} rows with missing data")
print(f"Missing values reduced from {missing_before} to {missing_after}")
print("Note: Removing rows with missing data may disproportionately exclude certain communities")

# Step 5: Filter for Disability Estimates - Explaining the categorization choices
# The classification system itself reflects institutional choices that may not capture everyone's experience
disability_categories = [
    "Cognitive Disability", "Mobility Disability", "Independent Living Disability",
    "Hearing Difficulty", "Vision Difficulty", "Any Disability", "No Disability"
]
filtered_df = df[df['DisabilityType'].isin(disability_categories)]
excluded_categories = df[~df['DisabilityType'].isin(disability_categories)]['DisabilityType'].unique()
print(f"I've included {len(disability_categories)} disability categories and excluded {len(excluded_categories)} others")
print(f"Excluded categories include: {', '.join(excluded_categories[:5])}{'...' if len(excluded_categories) > 5 else ''}")
print("This categorization reflects institutional definitions that may not align with lived experiences")

# Step 6: Identify Key Insights - Acknowledging temporal limitations
latest_year = filtered_df['Year'].max()
earliest_year = filtered_df['Year'].min()
year_range = latest_year - earliest_year
latest_df = filtered_df[filtered_df['Year'] == latest_year]
print(f"Data spans {year_range} years ({earliest_year}-{latest_year})")
print(f"Short time ranges may mask longer historical trends and structural factors")

# Step 7: Visualization 1 - Embracing complexity in trends over time
plt.figure(figsize=(14, 8))
sns.lineplot(data=filtered_df, x='Year', y='Rate', hue='DisabilityType', marker='o')

# Adding uncertainty bands to acknowledge variability
for category in disability_categories:
    category_data = filtered_df[filtered_df['DisabilityType'] == category]
    if not category_data.empty:
        # Calculate confidence interval (simplified approach)
        mean_by_year = category_data.groupby('Year')['Rate'].mean()
        std_by_year = category_data.groupby('Year')['Rate'].std().fillna(0)
        years = mean_by_year.index
        plt.fill_between(years, 
                         mean_by_year - std_by_year, 
                         mean_by_year + std_by_year, 
                         alpha=0.2)

plt.title(f'Trend of Disability Types in US (Age-adjusted Prevalence) - {earliest_year} to {latest_year}')
plt.ylabel('Rate (%) - Note: Variations may reflect changes in reporting methods')
plt.grid(True, alpha=0.3)
plt.legend(title='Disability Type')
plt.figtext(0.5, 0.01, 'Source: CDC - Variability bands represent state-level differences, not statistical confidence intervals', 
            ha='center', fontsize=9, style='italic')
plt.show()

# Step 8: Visualization 2 - Boxplot showing distribution complexity
plt.figure(figsize=(14, 7))
boxplot = sns.boxplot(x='DisabilityType', y='Rate', data=latest_df, palette='Set3')

# Adding data points to show actual distribution
sns.stripplot(x='DisabilityType', y='Rate', data=latest_df, 
              size=4, color=".3", alpha=0.5)

plt.title(f'Disability Rates by Type in {latest_year} - Showing Full Distribution')
plt.xticks(rotation=45)
plt.ylabel('Rate (%)')
plt.figtext(0.5, 0.01, 'Note: Each point represents a state - outliers may indicate unique regional factors worth investigating', 
            ha='center', fontsize=9, style='italic')
plt.show()

# Step 9: Visualization 3 - Top 5 States, but with context
# Focus on the states with the highest rates for "Any Disability."
top5_states = latest_df[latest_df['DisabilityType'] == 'Any Disability'].sort_values(by='Rate', ascending=False).head(5)
bottom5_states = latest_df[latest_df['DisabilityType'] == 'Any Disability'].sort_values(by='Rate').head(5)

# Combine for comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 5
sns.barplot(x='State', y='Rate', data=top5_states, palette='magma', ax=ax1)
ax1.set_title(f'Top 5 States by Any Disability Rate in {latest_year}')
ax1.set_ylabel('Rate (%)')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)

# Bottom 5
sns.barplot(x='State', y='Rate', data=bottom5_states, palette='magma', ax=ax2)
ax2.set_title(f'Bottom 5 States by Any Disability Rate in {latest_year}')
ax2.set_ylabel('Rate (%)')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)

plt.figtext(0.5, 0.01, 'Note: State rankings may correlate with socioeconomic factors, healthcare access, and reporting methods', 
            ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.show()

# Step 10: Visualization 4 - Enhanced Heatmap with critical context
# This heatmap shows the prevalence of each disability type across states in the latest year.
heatmap_df = latest_df.pivot(index='State', columns='DisabilityType', values='Rate').fillna(0)

# Adding state demographic data (simplified placeholder - in real analysis, would join actual data)
# This acknowledges that disability rates correlate with other factors
heatmap_df = heatmap_df.reset_index()
heatmap_df = heatmap_df.sort_values(by='Any Disability', ascending=False)

plt.figure(figsize=(16, 12))
sns.heatmap(heatmap_df.set_index('State'), cmap='YlGnBu', annot=True, fmt=".1f")
plt.title(f'Disability Rates Heatmap by State and Type in {latest_year}')
plt.ylabel('State (sorted by "Any Disability" rate)')
plt.xlabel('Disability Type')
plt.figtext(0.5, 0.01, 
            'Note: State rankings may reflect differences in demographics, healthcare systems, and economic conditions\n'
            'rather than inherent differences in disability prevalence. This data should inform support systems, not comparisons.',
            ha='center', fontsize=9, style='italic')
plt.show()

# Step 11: Critical Questions - Adding analytical reflection
print("\n===== MY CRITICAL REFLECTIONS ON THIS DATA =====")
print("1. Who benefits from measuring disability this way? Federal agencies, healthcare systems, advocacy groups.")
print("2. Who might be harmed? People with disabilities that don't fit neatly into these categories.")
print("3. What remains unmeasured? Severity, intersectionality with other identities, personal experiences.")
print("4. How might self-reporting bias affect this data? Stigma and access issues may lead to underreporting.")
print("5. Economic context: How do economic opportunities correlate with these rates?")
print("6. Healthcare access: How do healthcare disparities manifest in this data?")

# A note on limitations
print("\n===== THE LIMITATIONS =====\n"
      "This analysis provides insights into disability rates across the United States, "
      "but I recognize the limitations of categorical data in representing human experiences. "
      "I've embraced complexity where possible and tried to acknowledge what remains unmeasured. "
      "This data should inform supportive policies and resource allocation, rather than "
      "reductive comparisons between states or disability types. "
      "My goal is to create analysis that returns value to the communities represented in this data.")
