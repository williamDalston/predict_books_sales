# Get book data
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup

# Set the URL of the Amazon Best Sellers page for the desired category
url = "https://www.amazon.com/best-sellers-books"

# Send a request to the URL and retrieve the HTML content
response = requests.get(url)
html = response.content

# Use BeautifulSoup to parse the HTML content and extract the relevant data
soup = BeautifulSoup(html, 'html.parser')
books = soup.find_all('div', class_='a-section a-spacing-none p13n-asin')

# Loop through the books and extract the book title, author, and rank
for book in books:
    title = book.find('span', class_='a-size-medium a-color-base a-text-normal').text.strip()
    author = book.find('a', class_='a-size-base a-link-normal').text.strip()
    rank = book.find('span', class_='zg-badge-text').text.strip()

    # Print the book title, author, and rank
    print(title, author, rank)


# Create a list of dictionaries to store the book data
books_list = []

# Loop through the books and extract the book title, author, and rank
for book in books:
    title = book.find('span', class_='a-size-medium a-color-base a-text-normal').text.strip()
    author = book.find('a', class_='a-size-base a-link-normal').text.strip()
    rank = book.find('span', class_='zg-badge-text').text.strip()

    # Add the book data to the list of dictionaries
    books_list.append({'Title': title, 'Author': author, 'Rank': rank})

# Convert the list of dictionaries to a Pandas DataFrame
books_df = pd.DataFrame(books_list)

# Remove any duplicate rows from the DataFrame
books_df.drop_duplicates(inplace=True)

# Convert the 'Rank' column to a numerical data type
books_df['Rank'] = pd.to_numeric(books_df['Rank'])

# Sort the DataFrame by rank in ascending order
books_df.sort_values(by='Rank', inplace=True)

# Reset the index of the DataFrame
books_df.reset_index(drop=True, inplace=True)

# Print the cleaned and processed data
print(books_df.head())


# Load the book sales data into a Pandas DataFrame
books_df = pd.read_csv('book_sales_data.csv')

# Calculate the total sales volume for each book genre
sales_by_genre = books_df.groupby('Genre')['Sales Volume'].sum()

# Calculate the average rating for each book genre
ratings_by_genre = books_df.groupby('Genre')['Average Rating'].mean()

# Calculate the percentage of sales for each book genre
total_sales = sales_by_genre.sum()
sales_percentages = sales_by_genre / total_sales * 100

# Print the results of the analysis
print("Total Sales by Genre:\n", sales_by_genre)
print("Average Ratings by Genre:\n", ratings_by_genre)
print("Sales Percentages by Genre:\n", sales_percentages)


# Load the book sales data into a Pandas DataFrame
books_df = pd.read_csv('book_sales_data.csv')

# Split the data into training and testing sets
train_data = books_df.iloc[:-1]
test_data = books_df.iloc[-1:]

# Select the features and target variable for the model
features = ['Sales Volume']
target = 'Month'

# Fit a linear regression model to the training data
lr_model = LinearRegression()
lr_model.fit(train_data[features], train_data[target])

# Use the model to predict sales for the next month
next_month_sales = lr_model.predict(test_data[features])

# Print the predicted sales for the next month
print("Predicted sales for next month:", next_month_sales)
