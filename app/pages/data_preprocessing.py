# app/pages/data_preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
import clustering
import correlation
import data_preprocessing
import data_transformation
import dimensionality_reduction
import feature_engineering
import datetime
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


def clean_date_column(df):
    """Clean the Date column by removing timezone information and standardizing format"""
    if 'Date' in df.columns:
        # Convert to datetime first
        df['Date'] = pd.to_datetime(df['Date'])

        # Remove timezone information
        if hasattr(df['Date'].dt, 'tz'):
            df['Date'] = df['Date'].dt.tz_localize(None)

        # Standardize date format with no time component
        df['Date'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d'))

    return df


def data_preprocessing_page():
    st.header("Cryptocurrency Data Analysis")

    # Sidebar for CSV selection
    st.sidebar.header('Data Selection')

    # Get all CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

    if not csv_files:
        st.sidebar.warning("No CSV files found. Please add CSV files to the current directory.")
    else:
        # Select a CSV file with a more descriptive label
        selected_csv = st.sidebar.selectbox("Select cryptocurrency data file:", options=csv_files)

        # Add instructions
        st.sidebar.info("Click 'Load Data' to process the selected CSV file.")

        # Button to load the data
        load_button = st.sidebar.button('Load Data')

    # Handle load button click
    if 'load_button' in locals() and load_button and selected_csv:
        with st.spinner('Loading and analyzing data...'):
            try:
                # Load the CSV file
                file_path = os.path.join('.', selected_csv)
                df = pd.read_csv(file_path)

                if df.empty:
                    st.error("The selected CSV file is empty.")
                else:
                    # Clean the date column to remove timezone information
                    df = clean_date_column(df)

                    # Store in session state
                    st.session_state.crypto_data = df

                    # Display basic info
                    st.success(f"Successfully loaded {len(df)} rows from {selected_csv}")

                    # Display dataset summary
                    st.subheader("Dataset Summary")
                    col1, col2, col3 = st.columns(3)

                    # Process date information if available
                    if 'Date' in df.columns:
                        min_date = df['Date'].min()
                        max_date = df['Date'].max()
                        date_range = (max_date - min_date).days

                        with col1:
                            st.metric("Date Range", f"{date_range} days")
                            st.write(f"From: {min_date.date()}")
                            st.write(f"To: {max_date.date()}")

                    # Get cryptocurrency information
                    if 'Ticker' in df.columns:
                        unique_tickers = df['Ticker'].unique()
                        with col2:
                            st.metric("Cryptocurrencies", len(unique_tickers))
                            st.write("Tickers:")
                            st.write(", ".join(unique_tickers[:10]) +
                                     ("..." if len(unique_tickers) > 10 else ""))

                    # Calculate data points
                    with col3:
                        data_points = len(df)
                        st.metric("Data Points", f"{data_points:,}")
                        if 'Close' in df.columns:
                            avg_price = df['Close'].mean()
                            st.write(f"Avg Price: ${avg_price:.2f}")

                    # Display sample data
                    st.subheader("Data Sample")
                    st.dataframe(df.head())

                    # Show basic statistics
                    if 'Close' in df.columns and 'Ticker' in df.columns and 'Date' in df.columns:
                        st.subheader("Price Statistics by Cryptocurrency")
                        price_stats = df.groupby('Ticker')['Close'].agg(['mean', 'min', 'max', 'std']).reset_index()
                        price_stats.columns = ['Ticker', 'Mean Price', 'Min Price', 'Max Price', 'Price Std Dev']
                        price_stats = price_stats.round(2)
                        st.dataframe(price_stats)
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.info(
                    "Make sure the CSV file has the correct structure with columns like 'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'.")

    # Data processing section
    if 'crypto_data' in st.session_state and not st.session_state.crypto_data.empty:
        st.markdown("---")
        st.header("Data Preprocessing and Transformation")

        # Make a copy to avoid modifying the original
        df = st.session_state.crypto_data.copy()

        # Ensure Date is available as a column
        if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            st.write("Reset DateTime index to 'Date' column.")
        elif 'date' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'date': 'Date'})
            st.write("Renamed 'date' column to 'Date'.")

        # Ensure Date is datetime type with no timezone
        if 'Date' in df.columns:
            df = clean_date_column(df)
            st.write("Cleaned and standardized Date column format.")

        # Remove duplicates
        df = data_transformation.remove_duplicates(df, subset=['Ticker', 'Date'])
        st.write("Duplicates removed.")

        # Continue with the rest of your processing
        try:
            # Pivot and fill data
            processed_data = data_transformation.pivot_and_fill(df, index='Ticker', columns='Date', values='Close')
            st.write("Data pivoted and filled.")
            st.dataframe(processed_data.head())

            # Dynamic PCA components calculation
            n_samples = len(processed_data)  # Number of cryptocurrencies
            n_features = processed_data.shape[1]  # Number of time points
            max_possible = min(n_samples, n_features) - 1
            n_components = min(10, max_possible)

            st.write(f"Using {n_components} PCA components (based on data size)")

            scaled_data = data_transformation.scale_data(processed_data)
            st.write("Data scaled.")

            reduced_data_df, pca = dimensionality_reduction.apply_pca(scaled_data, n_components=n_components)
            st.write("PCA applied.")
            st.dataframe(reduced_data_df.head())

            # Display explained variance plot
            fig = dimensionality_reduction.plot_explained_variance(pca, n_components=n_components)
            st.pyplot(fig)

            # Determine optimal number of clusters (between 2 and 4)
            n_clusters = min(4, n_samples)
            if n_clusters < 2:
                n_clusters = 2

            st.write(f"Using {n_clusters} clusters for K-means")

            # Perform clustering
            clusters, kmeans = clustering.apply_kmeans(reduced_data_df, n_clusters=n_clusters)
            processed_data = clustering.add_cluster_labels(processed_data, clusters)
            st.write("K-means clustering applied.")
            st.dataframe(processed_data.head())

            # Show the counts of cryptocurrencies in each cluster
            st.write("Cluster Distribution:")
            st.write(processed_data['Cluster'].value_counts())

            # Visualize clusters
            fig = clustering.plot_clusters(reduced_data_df.values, clusters)
            st.pyplot(fig)

            # Automatically select cryptos closest to centroids
            selected_cryptos = clustering.select_cryptos_closest_to_centroids(reduced_data_df, clusters,
                                                                              kmeans.cluster_centers_)
            st.write("Cryptocurrencies closest to each cluster centroid:")
            st.dataframe(selected_cryptos)

            # Convert 'Date' column to datetime and set as index
            df = data_preprocessing.convert_to_datetime(df, 'Date').set_index('Date')

            # Drop every other column on selected cryptos, except the Ticker and cluster
            selected_cryptos = selected_cryptos[['Cluster']]
            st.write("Selected Cryptos:")
            st.write(selected_cryptos)

            # Merge the selected cryptocurrencies with the original data to get their full details including Date
            selected_cryptos_full = df.merge(selected_cryptos, left_on='Ticker', right_index=True)

            # Make sure the data is sorted by Date and its a column not an index
            selected_cryptos_full = selected_cryptos_full.reset_index().sort_values('Date')
            st.write("Selected Cryptos Full Details:")
            st.write(selected_cryptos_full)

            # Calculate daily returns for all cryptocurrencies
            daily_returns = correlation.calculate_daily_returns(df)

            # Apply the feature engineering to the selected cryptos
            selected_cryptos_full = feature_engineering.calculate_sma_ema_rsi(selected_cryptos_full)
            st.write("Selected Cryptos Full Details with Feature Engineering:")
            st.write(selected_cryptos_full)

            # Convert to datetime and set as index in selected cryptos full
            selected_cryptos_full = data_preprocessing.convert_to_datetime(selected_cryptos_full, 'Date').set_index(
                'Date')
            st.write("Selected Cryptos Full Details with Feature Engineering with Date as Index:")
            st.write(selected_cryptos_full)

            # Store in session state
            st.session_state['selected_cryptos'] = selected_cryptos
            st.session_state['daily_returns'] = daily_returns
            st.session_state['selected_cryptos_full'] = selected_cryptos_full

            st.success("Preprocessing completed. Data stored in session state and ready for analysis.")

            # Add helpful explanation of results
            st.info("""
            **What's happening:**
            1. Your cryptocurrency data has been processed and analyzed
            2. We've identified clusters of cryptocurrencies with similar price movements
            3. For each cluster, we've selected the most representative cryptocurrency
            4. These selected cryptocurrencies can be used as proxies for their respective clusters
            5. Technical indicators (SMA, EMA, RSI) have been calculated for deeper analysis
            """)

        except Exception as e:
            st.error(f"Error during data processing: {e}")
            st.info("Check the data structure and try again.")
    else:
        st.info("Please load a CSV file to proceed with transformation and analysis.")