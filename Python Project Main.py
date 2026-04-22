import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv(r"C:\Users\himan\Downloads\Car .xlsx - car_data.csv")

print("First 5 Rows of Dataset: ", df.head())
print("Columns:\n", df.columns)


# Rename column for easy use
df.rename(columns={'Price ($)': 'Price'}, inplace=True)

# Convert to numeric
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Convert Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Handle missing values
df['Price'] = df['Price'].ffill()

df = df.dropna()

print("Missing Values:\n", df.isnull().sum())

# Average price by company
company_price = df.groupby('Company')['Price'].mean()

# Average price by body style
body_style_price = df.groupby('Body Style')['Price'].mean()

# Cars with high price
expensive_cars = df[df['Price'] > df['Price'].mean()]

# Q1: How does car price vary over time?
plt.figure(figsize=(12,8))
plt.plot(df['Date'], df['Price'], color='violet', linewidth=2, marker='d')
plt.title("Price Trend Over Time", fontsize=14, color='black')
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Q2: Which company offers the most expensive cars?
company_price.plot(
    kind='bar',
    figsize=(10,5),
    color='orange',
    edgecolor='black'
)

plt.title("Average Price by Company", fontsize=14, color='darkred')
plt.xlabel("Company")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Q3: What is the distribution of car prices?
plt.figure(figsize=(8,5))
plt.hist(df['Price'], bins=20, color='purple', edgecolor='black')
plt.title("Price Distribution", fontsize=14, color='purple')
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Q4: Are there any outliers in the data?
'''plt.figure(figsize=(8,5))
sns.boxplot(x=df['Price'], color='cyan')
plt.title("Outlier Detection (Price)", fontsize=14, color='darkgreen')
plt.show() 
'''
plt.figure(figsize=(5,8))
sns.boxplot(y=df['Price'], color='cyan')
plt.title("Outlier Detection (Price)", fontsize=14, color='darkgreen')
plt.show()

# Q5: What is the relationship between income and price?
corr = df[['Price', 'Annual Income']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=1, linecolor='black')
plt.title("Correlation Heatmap", fontsize=14, color='brown')
plt.show()

# Q6: What is the distribution of customers based on Gender?
gender_counts = df['Gender'].value_counts()
plt.figure()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.show()


# Q7: Represent Gender distribution using a Donut Chart
plt.figure()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
centre_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title("Gender Distribution (Donut)")
plt.show()

# Q8: Which car companies are most preferred by customers?
company_counts = df['Company'].value_counts().head(5)
plt.figure()
plt.pie(company_counts, labels=company_counts.index, autopct='%1.1f%%')
plt.title("Top 5 Companies")
plt.show()

# Q9: Show top car companies using Donut Chart
plt.figure()
plt.pie(company_counts, labels=company_counts.index, autopct='%1.1f%%')
centre_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title("Top Companies (Donut)")
plt.show()


# Q10: What is the preferred transmission type among customers?
transmission_counts = df['Transmission'].value_counts()
plt.figure()
plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%')
plt.title("Transmission Type")
plt.show()


# Q11: Which body style of cars is most popular?
body_counts = df['Body Style'].value_counts()
plt.figure()
plt.pie(body_counts, labels=body_counts.index, autopct='%1.1f%%')
plt.title("Body Style Distribution")
plt.show()

# Q12: Which dealer region has the highest number of customers?
region_counts = df['Dealer_Region'].value_counts()
plt.figure()
plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
plt.title("Dealer Region Distribution")
plt.show()

# Q13: What is the relationship between customer income and car price?
sns.scatterplot(x=df['Price'], y=df['Annual Income'])
plt.title("Income vs Price")
plt.show()


# Summary stats
print(df.describe())

# Correlation
print("Correlation:\n", df[['Price','Annual Income']].corr())

# Covariance
print("Covariance:\n", df[['Price','Annual Income']].cov())


arr = np.array(df['Price'])

print("Mean Price:", np.mean(arr))
print("Max Price:", np.max(arr))
print("Standard Deviation:", np.std(arr))

print("Project Completed Successfully ")
