import yfinance as yf

ticker = 'AAPL'  
data = yf.download(ticker, start='2020-01-01', end='2025-01-23')

data.to_csv('historical_stock_data.csv')

print("Data downloaded and saved to historical_stock_data.csv")