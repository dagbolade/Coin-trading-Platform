import requests
import streamlit as st


# API key here
API_KEY = 'pub_3971050fd96dbc2e856ed4aa8aeaa79c531bb'


def fetch_news(api_key, query):
    url = f'https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data['results'] if 'results' in news_data else []
    else:
        st.error(f'Failed to fetch news: {response.status_code}')
        return []


def display_news(articles):
    if articles:
        for article in articles:
            st.subheader(article['title'] if 'title' in article else 'No Title')
            st.write(article['content'] if 'content' in article else 'No Content Available')
            if 'link' in article and article['link']:
                st.markdown(f"[Read more]({article['link']})", unsafe_allow_html=True)
            st.write("---")  # For a separator line
    else:
        st.error('No news articles found.')


def news_page():
    st.info("This page allows you to search for news articles related to cryptocurrencies.")
    search_query = st.text_input("Search for news on any topic:")
    search_button = st.button('Search')

    if search_button and search_query:
        articles = fetch_news(API_KEY, search_query)
        display_news(articles)
    elif search_button:
        st.error("Please enter a search term.")

    # Check if selected cryptos are available
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        st.info("You can also select a cryptocurrency to fetch news related to it.")
        ticker = st.selectbox('Or select a Cryptocurrency for news:', unique_tickers, index=0)
        fetch_button = st.button(f"Fetch news for {ticker}")

        if fetch_button:
            articles = fetch_news(API_KEY, ticker)
            display_news(articles)
    else:
        st.error("Cryptocurrency data is not loaded. Please load the data to proceed.")