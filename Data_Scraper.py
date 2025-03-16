# scrape data from yfinance for 30 cryptocurrencies
from datetime import datetime

import pandas as pd
import yfinance as yf
import json


def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for stock: {ticker}")
        print(e)
        return pd.DataFrame()


def save_to_json(data, group):
    # Ensuring the 'Ticker' is present in the data
    if 'Ticker' not in data.columns:
        data['Ticker'] = data.index

    # Use DateTime index as 'Date' column
    data['Date'] = data.index

    # Convert the data to a dictionary
    data_dict = data.to_dict(orient='split')


# List of cryptocurrencies to fetch data for
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD',
                    'BNB-USD', 'USDT-USD', 'EOS-USD', 'BSV-USD', 'XLM-USD',
                    'ADA-USD', 'TRX-USD', 'XTZ-USD', 'LINK-USD', 'XMR-USD',
                    'NEO-USD', 'LEO-USD', 'HT-USD', 'MIOTA-USD', 'ATOM1-USD',
                    'VET-USD', 'MKR-USD', 'CRO-USD', 'DAI-USD', 'ONT-USD',
                    'DOGE-USD', 'ZEC-USD', 'BAT-USD', 'DCR-USD', 'QTUM-USD']

# Fetch data for each cryptocurrency
for idx, crypto in enumerate(cryptocurrencies, 1):
    data = fetch_data(crypto, start_date='2023-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    save_to_json(data, idx)

# save to csv
data.to_csv('crypto_data.csv')
print('Data saved to crypto_data.csv')
