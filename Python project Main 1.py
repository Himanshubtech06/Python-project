import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

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
# Chart: Line Chart with Area Fill

df_monthly = df.set_index('Date')['Price'].resample('ME').mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#EEF2FF')
ax.set_facecolor('#EEF2FF')
ax.plot(df_monthly['Date'], df_monthly['Price'], color='violet', linewidth=2.5, marker='o',
        markersize=6, markerfacecolor='white', markeredgecolor='violet', markeredgewidth=2)
ax.fill_between(df_monthly['Date'], df_monthly['Price'], alpha=0.25, color='violet')
ax.set_title("Price Trend Over Time (Monthly Average)", fontsize=14, color='indigo', pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Average Price ($)", fontsize=12)
plt.xticks(rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Q2: Which company offers the most expensive cars?
# Chart: Horizontal Bar Chart

top_companies = company_price.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#FFF8EC')
ax.set_facecolor('#FFF8EC')
colors = ['#FF8C00', '#FFA500', '#FFB732', '#FFC84A', '#FFD966',
          '#FFE57F', '#FFED99', '#FFF0B0', '#FFF4C4', '#FFF8D8']
bars = ax.barh(top_companies.index[::-1], top_companies.values[::-1],
               color=colors, edgecolor='black', height=0.6)
for bar, val in zip(bars, top_companies.values[::-1]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}', va='center', fontsize=10, color='black')
ax.set_title("Average Price by Company (Top 10)", fontsize=14, color='darkred', pad=15)
ax.set_xlabel("Average Price ($)", fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Q3: What is the distribution of car prices?
# Chart: Histogram with KDE Curve and Mean Line

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#F5EEFF')
ax.set_facecolor('#F5EEFF')
ax.hist(df['Price'], bins=30, color='mediumpurple', edgecolor='black', density=True, alpha=0.75)
kde_x = np.linspace(df['Price'].min(), df['Price'].max(), 300)
kde_y = gaussian_kde(df['Price'])(kde_x)
ax.plot(kde_x, kde_y, color='red', linewidth=2.5, label='KDE Curve')
ax.axvline(df['Price'].mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Mean:  ${df["Price"].mean():,.0f}')
ax.set_title("Price Distribution with KDE", fontsize=14, color='purple', pad=15)
ax.set_xlabel("Price ($)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Q4: Are there any outliers in the data?
# Chart: Boxplot

fig, ax = plt.subplots(figsize=(5, 8))
fig.patch.set_facecolor('#E8FFF5')
ax.set_facecolor('#E8FFF5')
sns.boxplot(y=df['Price'], color='cyan', ax=ax,
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='darkgreen', linewidth=1.5),
            capprops=dict(color='darkgreen', linewidth=2),
            boxprops=dict(edgecolor='darkgreen', linewidth=1.5),
            flierprops=dict(marker='o', color='red', markersize=4, alpha=0.5))
ax.set_title("Outlier Detection (Price)", fontsize=14, color='darkgreen', pad=15)
ax.set_ylabel("Price ($)", fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Q5: What is the relationship between income and price?
# Chart: Correlation Heatmap

corr = df[['Price', 'Annual Income']].corr()

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('#FFF0F0')
ax.set_facecolor('#FFF0F0')
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
            linewidths=1.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
ax.set_title("Price vs Annual Income", fontsize=13, color='brown', pad=15)
plt.tight_layout()
plt.show()



# Q6: What is the distribution of customers based on Gender?
# Chart: Pie Chart

gender_counts = df['Gender'].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#EEF6FF')
wedges, texts, autotexts = ax.pie(
    gender_counts, labels=gender_counts.index,
    autopct='%1.1f%%', colors=['skyblue', 'pink'],
    startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for t in texts:      t.set_fontsize(12)
for at in autotexts: at.set(fontsize=11, fontweight='bold')
ax.set_title("Gender Distribution", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# Q7: Represent Gender distribution using a Donut Chart
# Chart: Donut Chart

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#EEF6FF')
wedges, texts, autotexts = ax.pie(
    gender_counts, labels=gender_counts.index,
    autopct='%1.1f%%', colors=['skyblue', 'pink'],
    startangle=140, pctdistance=0.82,
    wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2))
for t in texts:      t.set_fontsize(12)
for at in autotexts: at.set(fontsize=11, fontweight='bold')
centre_circle = plt.Circle((0, 0), 0.48, fc='#EEF6FF')
ax.add_artist(centre_circle)
ax.text(0, 0, f'{gender_counts.sum():,}\nCustomers',
        ha='center', va='center', fontsize=12, fontweight='bold', color='navy')
ax.set_title("Gender Distribution (Donut)", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q8: Which car companies are most preferred by customers?
# Chart: Pie Chart
# ─────────────────────────────────────────────────────────────────────────────
company_counts = df['Company'].value_counts().head(5)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#FFFBEE')
wedges, texts, autotexts = ax.pie(
    company_counts, labels=company_counts.index,
    autopct='%1.1f%%', startangle=120,
    colors=['#4F8EF7', '#F76F53', '#36C98E', '#F7C948', '#A78BFA'],
    explode=[0.04] * 5, pctdistance=0.80,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for t in texts:      t.set_fontsize(11)
for at in autotexts: at.set(fontsize=10, fontweight='bold')
ax.set_title("Top 5 Most Preferred Companies", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q9: Show top car companies using Donut Chart
# Chart: Donut Chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#FFFBEE')
wedges, texts, autotexts = ax.pie(
    company_counts, labels=company_counts.index,
    autopct='%1.1f%%', startangle=120,
    colors=['#4F8EF7', '#F76F53', '#36C98E', '#F7C948', '#A78BFA'],
    pctdistance=0.82,
    wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2))
for t in texts:      t.set_fontsize(11)
for at in autotexts: at.set(fontsize=10, fontweight='bold')
centre_circle = plt.Circle((0, 0), 0.48, fc='#FFFBEE')
ax.add_artist(centre_circle)
ax.text(0, 0, 'Top 5\nBrands', ha='center', va='center',
        fontsize=12, fontweight='bold', color='darkorange')
ax.set_title("Top Companies (Donut)", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q10: What is the preferred transmission type among customers?
# Chart: Donut Chart
# ─────────────────────────────────────────────────────────────────────────────
transmission_counts = df['Transmission'].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#EFFFEF')
wedges, texts, autotexts = ax.pie(
    transmission_counts, labels=transmission_counts.index,
    autopct='%1.1f%%', colors=['lightgreen', 'gold'],
    startangle=90, pctdistance=0.82,
    wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2))
for t in texts:      t.set_fontsize(12)
for at in autotexts: at.set(fontsize=11, fontweight='bold')
centre_circle = plt.Circle((0, 0), 0.48, fc='#EFFFEF')
ax.add_artist(centre_circle)
ax.text(0, 0, 'Trans.\nType', ha='center', va='center',
        fontsize=12, fontweight='bold', color='darkgreen')
ax.set_title("Preferred Transmission Type", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q11: Which body style of cars is most popular?
# Chart: Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
body_counts = df['Body Style'].value_counts()

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#E8F4FF')
ax.set_facecolor('#E8F4FF')
bar_colors = ['#4F8EF7', '#36C98E', '#F7C948', '#F76F53', '#A78BFA']
bars = ax.bar(body_counts.index, body_counts.values,
              color=bar_colors, edgecolor='black', width=0.55)
for bar, val in zip(bars, body_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{val:,}', ha='center', fontsize=10, color='black')
ax.set_title("Body Style Popularity", fontsize=14, color='navy', pad=15)
ax.set_xlabel("Body Style", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q12: Which dealer region has the highest number of customers?
# Chart: Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
region_counts = df['Dealer_Region'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#FFF0E8')
ax.set_facecolor('#FFF0E8')
reg_colors = ['#FF6B6B', '#FF8E53', '#FFA500', '#FFB732', '#FFC84A', '#FFD966', '#FFE57F']
bars = ax.bar(region_counts.index, region_counts.values,
              color=reg_colors, edgecolor='black', width=0.55)
for bar, val in zip(bars, region_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{val:,}', ha='center', fontsize=9.5, color='black')
ax.set_title("Customers by Dealer Region", fontsize=14, color='darkred', pad=15)
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=20, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q13: What is the relationship between customer income and car price?
# Chart: Scatter Plot
# ─────────────────────────────────────────────────────────────────────────────
sample = df[['Price', 'Annual Income']].sample(n=min(500, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#E8FFFA')
ax.set_facecolor('#E8FFFA')
scatter = ax.scatter(sample['Price'], sample['Annual Income'],
                     c=sample['Price'], cmap='cool',
                     alpha=0.65, s=40, edgecolors='none')
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Price ($)', fontsize=11)
ax.set_title("Annual Income vs Car Price", fontsize=14, color='darkcyan', pad=15)
ax.set_xlabel("Car Price ($)", fontsize=12)
ax.set_ylabel("Annual Income ($)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q14: How is price spread across body styles?
# Chart: Violin Plot  (FutureWarning fixed using hue= and legend=False)
# ─────────────────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(11, 6))
# fig.patch.set_facecolor('#FFF5FF')
# ax.set_facecolor('#FFF5FF')
# sns.violinplot(x='Body Style', y='Price', hue='Body Style',
#                data=df, palette='pastel', legend=False, ax=ax)
# ax.set_title("Price Distribution by Body Style (Violin Plot)", fontsize=14, color='purple', pad=15)
# ax.set_xlabel("Body Style", fontsize=12)
# ax.set_ylabel("Price ($)", fontsize=12)
# ax.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q15: Monthly sales volume by Transmission type
# Chart: Stacked Area Chart
# ─────────────────────────────────────────────────────────────────────────────
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
monthly_trans = df.groupby(['Month', 'Transmission']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#E8F5FF')
ax.set_facecolor('#E8F5FF')
ax.stackplot(monthly_trans.index,
             monthly_trans['Auto'], monthly_trans['Manual'],
             labels=['Auto', 'Manual'],
             colors=['skyblue', 'lightgreen'], alpha=0.85)
ax.set_title(" Monthly Sales Volume by Transmission (Stacked Area)", fontsize=14, color='steelblue', pad=15)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Sales", fontsize=12)
ax.legend(loc='upper left', fontsize=10)
plt.xticks(rotation=35, ha='right')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Q18: Avg Income vs Avg Price by Company
# Chart: Bubble Chart
# ─────────────────────────────────────────────────────────────────────────────
top5 = df['Company'].value_counts().head(5).index.tolist()
bubble_data = df[df['Company'].isin(top5)].groupby('Company').agg(
    count=('Price', 'count'),
    avg_price=('Price', 'mean'),
    avg_income=('Annual Income', 'mean')
).reset_index()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#FFFAE8')
ax.set_facecolor('#FFFAE8')
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, row in bubble_data.iterrows():
    ax.scatter(row['avg_price'], row['avg_income'],
               s=row['count'] / 3,
               color=colors[i], alpha=0.7, edgecolors='black', linewidth=1.2)
    ax.text(row['avg_price'], row['avg_income'] + 10000,
            row['Company'], ha='center', fontsize=10, fontweight='bold', color=colors[i])
ax.set_title("Avg Income vs Avg Price by Company\n(Bubble size = number of sales)",
             fontsize=14, color='darkorange', pad=15)
ax.set_xlabel("Average Car Price ($)", fontsize=12)
ax.set_ylabel("Average Customer Income ($)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Summary stats
print(df.describe())

# Correlation
print("Correlation:\n", df[['Price', 'Annual Income']].corr())

# Covariance
print("Covariance:\n", df[['Price', 'Annual Income']].cov())

arr = np.array(df['Price'])

print("Mean Price:", np.mean(arr))
print("Max Price:", np.max(arr))
print("Standard Deviation:", np.std(arr))


