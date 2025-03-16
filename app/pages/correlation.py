# app/pages/correlation.py

import streamlit as st
import pandas as pd

import correlation


def correlation_page():
    st.header("Correlation Analysis")
    if 'selected_cryptos' in st.session_state and 'daily_returns' in st.session_state:
        selected_cryptos = st.session_state['selected_cryptos']
        daily_returns = st.session_state['daily_returns']

        st.write("Selected Cryptocurrencies:", selected_cryptos)
        st.write("Daily Returns Sample:", daily_returns.head())

        # Perform correlation analysis using these
        correlation_matrix = correlation.calculate_correlation_matrix(daily_returns)
        top_correlations = correlation.find_top_correlations(correlation_matrix, selected_cryptos.index.tolist())

        # Display top correlations
        for ticker, correlation_data in top_correlations.items():
            st.write(f"Top correlations for {ticker}")
            st.write("Positive Correlations:", correlation_data['Positive'])
            st.write("Negative Correlations:", correlation_data['Negative'])

    else:
        st.error("Data not available. Please run the preprocessing first.")