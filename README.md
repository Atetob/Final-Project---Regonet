# Final-Project---Regonet
By leveraging data-driven approaches, this hackathon aims to identify actionable insights, develop innovative solutions, and contribute to global efforts to reduce preventable deaths in children under five years of age.

import pandas as pd
import numpy as np

#Import Data as CSV
df1 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\1. youth-mortality-rate.csv")
print(df1)
df2 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\2. number-of-infant-deaths-unwpp.csv")
print(df2)
df3 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\3. child-mortality-by-income-level-of-country.csv")
print(df3)
df4 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\4. Distribution of Causes of Death among Children Aged less than 5 years.csv")
df4
df5 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\5. number-of-maternal-deaths-by-region.csv")
df5
df6 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\6. births-attended-by-health-staff-sdgs.csv")
df6
df7 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\7. global-vaccination-coverage.csv")
df7
df8 = pd.read_csv(r"C:\Users\oboma\Documents\Regonet Final Project\8. health-protection-coverage.csv")
df8

#Cleaning Data
##Remove Duplicates from Datasets
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()
df3 = df3.drop_duplicates()
df4 = df4.drop_duplicates()
df5 = df5.drop_duplicates()
df6 = df6.drop_duplicates()
df7 = df7.drop_duplicates()
df8 = df8.drop_duplicates()

##Check for missing Data and Remove them
print("Missing values:\\n", df1.isnull().sum())
df1 = df1.dropna(axis=0)
df1
print("Missing values:\\n", df2.isnull().sum())
df2 = df2.dropna(axis=0)
df2
print("Missing values:\\n", df3.isnull().sum())
df3 = df3.dropna(axis=0)
df3
print("Missing values:\\n", df4.isnull().sum())
df4 = df4.dropna(axis=1)
df4
print("Missing values:\\n", df5.isnull().sum())
df5 = df5[['Entity', 'Code', 'Year', 'Estimated maternal deaths']]
df5 = df5.dropna(axis=0)
df5
print("Missing values:\\n", df6.isnull().sum())
df6 = df6.dropna(axis=0)
df6
print("Missing values:\\n", df7.isnull().sum())
df7 = df7.fillna(0)
df7
print("Missing values:\\n", df8.isnull().sum())

##Modify Dataset 4 to unify with other Datasets
df4 = df4.rename(columns = {"SpatialDimValueCode" : "Code",
                     "Period":"Year",
                     "Location":"Entity",
                     })
df4

#Import Extra Libraries
import matplotlib.pyplot as plt
import seaborn as sns


#Filter Data for African Countries
def filter_african_countries(df):
     # Modify this list based on the actual data
    african_countries = ['Algeria', "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde", "Central Africa Republic", "Chad", "Congo", "Cote d'Ivoire", "Democratic Republic of Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "South Africa", "Somalia", "Sudan", "Tanzania", "Tunisia", "Uganda", "Zambia", "Zimbabwe"]
    return df[df["Entity"].isin(african_countries)]

df1 = filter_african_countries(df1)
df2 = filter_african_countries(df2)
df3 = filter_african_countries(df3)
df4 = filter_african_countries(df4)
df5 = filter_african_countries(df5)
df6 = filter_african_countries(df6)
df7 = filter_african_countries(df7)
df8 = filter_african_countries(df8)

# Merge datasets into a single DataFrame based on 'Entity', 'Code', and 'Year'
merged_df = df1.merge(df2, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df3, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df4, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df5, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df6, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df7, on=['Entity', 'Code', 'Year'], how='outer')
merged_df = merged_df.merge(df8, on=['Entity', 'Code', 'Year'], how='outer')

merged_df


#Fill call NaN values with 0
merged_df = merged_df.fillna(0)
merged_df

# Check data types of merged DataFrame
print(merged_df.dtypes)

#CHeck Again for Missing values
print("Missing values:\\n", merged_df.isnull().sum())


#Data Analysis

## Line Chart: Trends over time for Under-fifteen mortality rate
plt.figure(figsize=(10, 6))
for entity in merged_df['Entity'].unique():
    entity_df = merged_df[merged_df['Entity'] == entity]
    plt.plot(entity_df['Year'], entity_df['Under-fifteen mortality rate'], label=entity)
plt.title("Under-Fifteen Mortality Rate Trends")
plt.xlabel("Year")
plt.ylabel("Mortality Rate")
plt.legend()
plt.show()

## Pie Chart: Distribution of Health Insurance Coverage by Country
avg_insurance = merged_df.groupby('Entity')['Share of population covered by health insurance (ILO (2014))'].mean()

## Ensure the data is numeric and drop any invalid entries
avg_insurance = pd.to_numeric(avg_insurance, errors='coerce').dropna()

plt.figure(figsize=(8, 8))
plt.pie(avg_insurance, labels=avg_insurance.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Health Insurance Coverage")
plt.show()

## Heatmap: Focus on vaccination coverage
from folium.plugins import HeatMap

vaccination_columns = ['DTP3 (% of one-year-olds immunized)', 'BCG (% of one-year-olds immunized)',
                       'HepB3 (% of one-year-olds immunized)', 'MCV1 (% of one-year-olds immunized)']
vaccination_df = merged_df[['Entity'] + vaccination_columns]
vaccination_avg = vaccination_df.groupby('Entity').mean()
plt.figure(figsize=(10, 6))
sns.heatmap(vaccination_avg, annot=True, cmap="YlGnBu")
plt.title("Average Vaccination Coverage by Country")
plt.show()

##Correlation Analysis
### Encode 'Entity' as numeric for correlation analysis
merged_df['Entity_encoded'] = merged_df['Entity'].astype('category').cat.codes

### Select numeric columns including the encoded Entity
numeric_columns = merged_df.select_dtypes(include=['number'])

### Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

### Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Including Countries)")
plt.savefig('Correlation_Matrix_(Including Countries).png', format='png', dpi=400)
plt.show()

### Correlation heatmap for countries only
entity_corr = numeric_columns.corrwith(numeric_columns['Entity_encoded']).sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=entity_corr.index, y=entity_corr.values, palette="viridis")
plt.title("Correlation of Numeric Features with 'Entity' (Countries)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Correlation Coefficient")
plt.savefig('Correlation_of_Numeric_Features_with_Entity_(Countries).png', format='png', dpi=400)
plt.show()

## Mortality Rates vs. Vaccination Coverage
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="DTP3 (% of one-year-olds immunized)", y="Under-fifteen mortality rate", hue="Entity")
plt.title("Under-15 Mortality Rate vs. DTP3 Vaccination Coverage")
plt.xlabel("DTP3 Vaccination Coverage (%)")
plt.ylabel("Under-15 Mortality Rate")
plt.savefig('Mortality_Rates_vs._Vaccination_Coverage.png', format='png', dpi=400)
plt.show()

## Maternal Deaths vs. Births Attended by Skilled Staff
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="Births attended by skilled health staff (% of total)", y="Estimated maternal deaths", hue="Entity")
plt.title("Maternal Deaths vs. Skilled Birth Attendance")
plt.xlabel("Skilled Birth Attendance (%)")
plt.ylabel("Estimated Maternal Deaths")
plt.savefig('Maternal_Deaths_vs._Births_Attended_by_Skilled_Staff.png', format='png', dpi=400)
plt.show()

## Share of Population Covered by Health Insurance
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x="Entity", y="Share of population covered by health insurance (ILO (2014))")
plt.title("Health Insurance Coverage by Country")
plt.xlabel("Country")
plt.ylabel("Health Insurance Coverage (%)")
plt.xticks(rotation=45)
plt.savefig('Health_Insurance_Coverage_by_Country.png', format='png', dpi=400)
plt.show()

##Plot a Chloropleth Map
import plotly.express as px

### Ensure relevant columns are numeric (coerce errors to NaN)
for col in merged_df.columns:
    if col not in ['Entity', 'Code', 'Year']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

### Drop rows where 'Code' or 'Under-fifteen mortality rate' is missing
choropleth_data = merged_df[['Entity', 'Code', 'Under-fifteen mortality rate']].dropna()

### Group data to calculate the mean
choropleth_data = choropleth_data.groupby(['Entity', 'Code'], as_index=False).mean()

### Choropleth Map: Under-Fifteen Mortality Rate
fig = px.choropleth(
    data_frame=choropleth_data,
    locations='Code',  # ISO alpha-3 country codes
    color='Under-fifteen mortality rate',
    hover_name='Entity',
    title='Under-Fifteen Mortality Rate by Country',
    color_continuous_scale='Reds'
)

fig.show()


#Data Summary and Recommendation
columns_to_include = [
    'Under-fifteen mortality rate',
    'Estimated maternal deaths',
    'Births attended by skilled health staff (% of total)',
    'DTP3 (% of one-year-olds immunized)',
    'Share of population covered by health insurance (ILO (2014))'
]

for col in columns_to_include:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

cleaned_df = merged_df.dropna(subset=columns_to_include)

# Ensure numeric columns for grouping
summary = cleaned_df.groupby('Entity')[columns_to_include].apply(lambda x: x.astype(float).mean())
print("Summary of Key Indicators by Country")
print(summary)

summary.to_csv("healthcare_analysis_summary.csv")

recommendations = """
1. Increase vaccination coverage, particularly DTP3 and other critical vaccines, in countries with low coverage.
2. Improve access to skilled health professionals during childbirth through community training programs.
3. Expand health insurance coverage to ensure equitable access to healthcare services.
4. Develop targeted interventions to reduce maternal deaths by improving prenatal care and emergency response systems.
5. Conduct awareness campaigns in regions with high under-five mortality rates to encourage timely vaccination and maternal care.
"""

print("Recommendations:")
print(recommendations)
