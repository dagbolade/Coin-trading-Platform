#!/usr/bin/env python3
# standalone_crypto_downloader.py

import yfinance as yf
import pandas as pd
import datetime
import time
from tqdm import tqdm
import sys

# Set up console encoding for Windows
if sys.platform == 'win32':
    # Force UTF-8 encoding for console output
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class CryptoDataDownloader:
    def __init__(self, tickers, start_date=None, end_date=None, interval='1d'):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def download_data(self):
        all_crypto_data = pd.DataFrame()

        if self.start_date is None or self.end_date is None:
            self.start_date = datetime.datetime.now() - datetime.timedelta(days=365)  # Default to last year
            self.end_date = datetime.datetime.now()

        print(f"Downloading data from {self.start_date} to {self.end_date}")

        # Convert dates to string format if they are datetime objects
        start_date_str = self.start_date.strftime('%Y-%m-%d') if isinstance(self.start_date,
                                                                            datetime.datetime) else self.start_date
        end_date_str = self.end_date.strftime('%Y-%m-%d') if isinstance(self.end_date,
                                                                        datetime.datetime) else self.end_date

        for ticker in tqdm(self.tickers, desc="Downloading crypto data"):
            try:
                print(f"Downloading data for {ticker}")
                # Just use the direct ticker - simpler approach to avoid encoding issues
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start_date_str, end=end_date_str, interval=self.interval)

                if not data.empty:
                    print(f"Successfully downloaded data for {ticker}")
                    # Handle column case inconsistencies
                    data.columns = [col.capitalize() for col in data.columns]

                    # Filter out unwanted columns like 'Dividends' and 'Stock splits'
                    columns_to_remove = ['Dividends', 'Stock Splits', 'Dividend', 'Split']
                    for col in columns_to_remove:
                        if col in data.columns:
                            data = data.drop(columns=[col])

                    # Add ticker column - use the original ticker for consistency
                    data['Ticker'] = ticker

                    # Handle index formatting
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.index = pd.to_datetime(data.index)

                    # Append to master dataframe
                    all_crypto_data = pd.concat([all_crypto_data, data], ignore_index=False)

                else:
                    print(f"No data available for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")

            # Add a small delay between requests
            time.sleep(0.5)

        if all_crypto_data.empty:
            print("WARNING: No data was successfully downloaded for any of the tickers.")
        else:
            print(f"Successfully downloaded data for {len(all_crypto_data['Ticker'].unique())} cryptocurrencies.")

        return all_crypto_data

    def save_to_csv(self, dataframe, file_path):
        try:
            # Reset index to make sure Date is a column
            if isinstance(dataframe.index, pd.DatetimeIndex):
                dataframe = dataframe.reset_index()

            dataframe.to_csv(file_path, index=False, encoding='utf-8')
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            # Try alternative approach
            try:
                print("Trying alternative save method...")
                dataframe.to_csv(file_path, index=False, encoding='cp1252')
                print(f"Data successfully saved to {file_path} using cp1252 encoding")
            except Exception as e2:
                print(f"Alternative save also failed: {str(e2)}")


if __name__ == "__main__":
    # Define the list of cryptocurrency tickers
    crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
        'DOGE-USD', 'AVAX-USD', 'SHIB-USD', 'DOT-USD', 'LINK-USD',
        'LTC-USD', 'UNI-USD', 'BCH-USD', 'XLM-USD', 'ETC-USD',
        'THETA-USD', 'FIL-USD', 'TRX-USD', 'EOS-USD', 'NEO-USD',
        'AAVE-USD', 'ATOM-USD', 'WIF-USD', 'BONK-USD', 'MANA-USD',
        'ENJ-USD', 'CHZ-USD', 'BAT-USD', 'SAND-USD', 'DGB-USD',

    ]

    # Get the current date for the filename
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Calculate the date range - 1 year back from today
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1095)

    # Create the downloader
    downloader = CryptoDataDownloader(
        tickers=crypto_tickers,
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )

    # Download the data
    print(f"Downloading data for {len(crypto_tickers)} cryptocurrencies...")
    try:
        historical_data = downloader.download_data()

        # Save to CSV with date in filename
        csv_file_path = f'crypto_data_{current_date}.csv'
        downloader.save_to_csv(historical_data, csv_file_path)

        print(f"Download completed. CSV saved as {csv_file_path}")
    except Exception as e:
        print(f"Error in main process: {str(e)}")