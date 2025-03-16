# app/pages/eda.py
import streamlit as st
import pandas as pd

import eda


def eda_page():
    st.header("Exploratory Data Analysis")
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    # Ensure the required data is available in the session state
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Convert 'Date' if it's not already the index
        if 'Date' in selected_cryptos_full.columns:
            selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'])
            selected_cryptos_full.set_index('Date', inplace=True)

        # Create a dropdown menu to select a cryptocurrency
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        selected_ticker = st.selectbox('Select a Cryptocurrency', unique_tickers)

        # Filter data for the selected cryptocurrency
        ticker_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == selected_ticker]

        # Display plots for the selected cryptocurrency
        if not ticker_data.empty:
            # describe the data
            st.write(f"Descriptive statistics for {selected_ticker}:")
            st.write(ticker_data.describe())

            fig = eda.plot_time_series(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = eda.plot_rolling_statistics(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = eda.plot_boxplot(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = eda.plot_candlestick(ticker_data, selected_ticker)
            st.plotly_chart(fig)

            fig = eda.plot_volatility_clustering(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = eda.plot_kde_of_closes(ticker_data, [selected_ticker])
            st.pyplot(fig)

            fig = eda.plot_candlestick_with_signals_and_ma(ticker_data, selected_ticker)
            st.plotly_chart(fig)
        else:
            st.error("No data available for the selected cryptocurrency.")
    else:
        st.info("Please perform data preprocessing to generate and select cryptocurrencies for analysis.")