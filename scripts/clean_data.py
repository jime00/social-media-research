import pandas as pd
import numpy as np

# 1. Load the dataset (Replace with your actual file name)
df = pd.read_csv('social_media_data.csv')

# 2. Duplicate Records: Remove exact copies
df = df.drop_duplicates()

# 3. Missing Values: Fill blank numbers with the median value
df['screen_time_hours'] = df['screen_time_hours'].fillna(df['screen_time_hours'].median())

# 4. Incorrect Data Types: Ensure numeric columns are stored as numbers
df['screen_time_hours'] = pd.to_numeric(df['screen_time_hours'], errors='coerce')
df['sleep_hours'] = pd.to_numeric(df['sleep_hours'], errors='coerce')

# 5. Inconsistent Formats: Strip extra spaces and make text lowercase
df['platform'] = df['platform'].str.strip().str.lower()

# 6. Inconsistent Categorical Values: Fix text typos and variations
df['platform'] = df['platform'].replace({'insta': 'instagram', 'fb': 'facebook'})

# 7. Invalid Values: Remove impossible logic (e.g., more than 24 hours in a day)
df = df[(df['screen_time_hours'] >= 0) & (df['screen_time_hours'] <= 24)]
df = df[(df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24)]

# 8. Outliers: Cap extreme high values using statistical bounds
Q1 = df['screen_time_hours'].quantile(0.25)
Q3 = df['screen_time_hours'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['screen_time_hours'] = np.where(df['screen_time_hours'] > upper_bound, upper_bound, df['screen_time_hours'])

# 9. Data Transformations: Group screen time into new buckets (Low, Moderate, High)
bins = [-1, 3, 7, 24]
labels = ['Low', 'Moderate', 'High']
df['usage_level'] = pd.cut(df['screen_time_hours'], bins=bins, labels=labels)

# Save your shiny new clean dataset!
df.to_csv('cleaned_social_media_data.csv', index=False)
print("Data cleaning complete! Saved as 'cleaned_social_media_data.csv'.")
