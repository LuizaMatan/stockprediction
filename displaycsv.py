import pandas as pd

# Read the CSV file and rename columns
data = pd.read_csv("historical_stock_data.csv", skiprows=2)
data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Convert 'Date' to datetime and display the first few rows
data['Date'] = pd.to_datetime(data['Date'])
print(data.head())
print(data.columns)
