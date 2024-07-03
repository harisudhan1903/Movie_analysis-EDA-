import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv('imdb_top_1000.csv')

# Display basic information
df.info()

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap='magma')
plt.show()

# Drop rows with missing 'Gross' values
df.dropna(subset=['Gross'], inplace=True)

# Fill missing 'Certificate' and 'Meta_score' with the mode
df['Certificate'] = df['Certificate'].fillna(df['Certificate'].mode()[0])
df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mode()[0])

# Convert 'Runtime' and 'Gross' to integer
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)
df['Gross'] = df['Gross'].str.replace(',', '').astype(int)

# Remove rows with 'Released_Year' as 'PG' and convert to integer
df = df[df['Released_Year'] != 'PG']
df['Released_Year'] = df['Released_Year'].astype(int)

# Drop unnecessary columns
df.drop(['Overview', 'Poster_Link'], axis=1, inplace=True)

# Display updated information
df.info()

# Set up the figure and axis for multiple plots
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Duration vs. Rating
sns.scatterplot(x='IMDB_Rating', y='Runtime', data=df, alpha=0.5, ax=axes[0, 0], color='purple')
axes[0, 0].set_title('IMDB_Rating vs. Runtime')

# Votes vs. Rating
sns.scatterplot(x='IMDB_Rating', y='No_of_Votes', data=df, alpha=0.5, ax=axes[0, 1], color='darkorange')
axes[0, 1].set_title('IMDB_Rating vs. No_of_Votes')

# Year vs. Rating
sns.scatterplot(x='IMDB_Rating', y='Released_Year', data=df, alpha=0.5, ax=axes[1, 0], color='blue')
axes[1, 0].set_title('IMDB_Rating vs. Released_Year')

# Meta_score vs. Rating
sns.scatterplot(x='IMDB_Rating', y='Meta_score', data=df, alpha=0.5, ax=axes[1, 1], color='green')
axes[1, 1].set_title('IMDB_Rating vs. Meta_score')

# Gross vs. Rating
sns.scatterplot(x='IMDB_Rating', y='Gross', data=df, alpha=0.5, ax=axes[2, 0], color='red')
axes[2, 0].set_title('IMDB_Rating vs. Gross')

# Correlation matrix
numeric_features = df[['IMDB_Rating', 'Runtime', 'No_of_Votes', 'Released_Year', 'Meta_score', 'Gross']]
correlation_matrix = numeric_features.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, ax=axes[2, 1], cmap='coolwarm')
axes[2, 1].set_title('Correlation Matrix')

plt.show()

# Regression plots
plt.figure(figsize=(12, 6))
sns.regplot(x='Gross', y='Meta_score', data=df, color='coral').set_title('Gross vs. IMDb Meta_score')
plt.show()

plt.figure(figsize=(12, 6))
sns.regplot(x='Gross', y='No_of_Votes', data=df, color='navy').set_title('Gross vs. No_of_Votes')
plt.show()

plt.figure(figsize=(12, 6))
sns.regplot(x='Gross', y='Runtime', data=df, color='green').set_title('Gross vs. Runtime')
plt.show()

plt.figure(figsize=(12, 6))
sns.regplot(x='Gross', y='Released_Year', data=df, color='purple').set_title('Gross vs. Year')
plt.show()

# Distribution of Ratings
Distribution_Rating = df['IMDB_Rating'].value_counts().sort_index()
plt.figure(figsize=(16, 8))
sns.barplot(x=Distribution_Rating.index, y=Distribution_Rating.values, palette='magma')
plt.title("Distribution of Rating")
plt.xlabel('IMDB_Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.grid(True, axis='y')
plt.show()

# Distribution of Meta_scores
Distribution_Meta = df['Meta_score'].value_counts().sort_index()
plt.figure(figsize=(16, 8))
sns.barplot(x=Distribution_Meta.index, y=Distribution_Meta.values, palette='magma')
plt.title("Distribution of Meta_score")
plt.xlabel('Meta_score')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.grid(True, axis='y')
plt.show()

# Gross trends over the years
fig = px.area(df, x='Released_Year', y='Gross', hover_data=['Gross', 'Released_Year'], color_discrete_sequence=['indigo'])
fig.update_layout(title='Gross trends over the years', xaxis_title="Year", yaxis_title="Gross")
fig.show()

# Runtime trends over the years
fig = px.area(df, x='Released_Year', y='Runtime', hover_data=['Runtime', 'Released_Year'], color_discrete_sequence=['darkgreen'])
fig.update_layout(title='Runtime trends over the years', xaxis_title="Year", yaxis_title="Runtime")
fig.show()

# 5 years most rated
mean_rating_by_year = df.groupby("Released_Year")['IMDB_Rating'].mean().sort_values(ascending=False).head(5)
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_rating_by_year.index, y=mean_rating_by_year.values, palette='rocket')
plt.title('Top 5 Years by Average IMDB Rating')
plt.xlabel('Released Year')
plt.ylabel('Average IMDB Rating')
plt.show()

# 5 years least rated
mean_rating_by_year = df.groupby("Released_Year")['IMDB_Rating'].mean().sort_values(ascending=True).head(5)
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_rating_by_year.index, y=mean_rating_by_year.values, palette='rocket')
plt.title('Bottom 5 Years by Average IMDB Rating')
plt.xlabel('Released Year')
plt.ylabel('Average IMDB Rating')
plt.show()

# Top 20
fig, axes = plt.subplots()
grouped = df.groupby("Series_Title")
mean = pd.DataFrame(grouped["IMDB_Rating"].mean())
mean1 = mean.sort_values("IMDB_Rating", ascending=False)
sns.barplot(y=mean1.index[:20], x=mean1.iloc[:20, 0].values, ax=axes, palette='coolwarm')
for container in axes.containers:
    axes.bar_label(container)
plt.ylabel('Movie')
plt.xlabel('IMDB_Rating')
plt.grid(True, axis='x')
plt.show()

# 20 movies least rated
least_rated_movies = df.groupby("Series_Title")['IMDB_Rating'].mean().sort_values(ascending=True).head(20)
plt.figure(figsize=(12, 10))
sns.barplot(y=least_rated_movies.index, x=least_rated_movies.values, palette='coolwarm')
plt.title('Bottom 20 Movies by IMDB Rating')
plt.xlabel('Average IMDB Rating')
plt.ylabel('Movie')
plt.show()

# 10 movies most voted
most_voted_movies = df.groupby("Series_Title")['No_of_Votes'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=most_voted_movies.index, y=most_voted_movies.values, palette='plasma')
plt.title('Top 10 Movies by Number of Votes')
plt.xlabel('Movie')
plt.ylabel('Average Number of Votes')
plt.xticks(rotation=90)
plt.show()

# 10 movies least voted
least_voted_movies = df.groupby("Series_Title")['No_of_Votes'].mean().sort_values(ascending=True).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=least_voted_movies.index, y=least_voted_movies.values, palette='plasma')
plt.title('Bottom 10 Movies by Number of Votes')
plt.xlabel('Movie')
plt.ylabel('Average Number of Votes')
plt.xticks(rotation=90)
plt.show()

# 10 movies with highest gross
highest_grossing_movies = df.groupby("Series_Title")['Gross'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=highest_grossing_movies.index, y=highest_grossing_movies.values, palette='plasma')
plt.title('Top 10 Movies by Gross')
plt.xlabel('Movie')
plt.ylabel('Average Gross')
plt.xticks(rotation=90)
plt.show()

# 10 movies with lowest gross
lowest_grossing_movies = df.groupby("Series_Title")['Gross'].mean().sort_values(ascending=True).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=lowest_grossing_movies.index, y=lowest_grossing_movies.values, palette='plasma')
plt.title('Bottom 10 Movies by Gross')
plt.xlabel('Movie')
plt.ylabel('Average Gross')
plt.xticks

fig = px.pie(df, values='IMDB_Rating', names='Genre', hover_data=['Genre'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Rating by Genres')
fig.show()
