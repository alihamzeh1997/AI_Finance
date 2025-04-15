# app.py - Main Streamlit Application with Alpha Vantage
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import os
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pandas_ta as ta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# Configuration and API Keys
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# OpenRouter client setup
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Page configuration
st.set_page_config(
    page_title="Financial Insights Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "daily_summaries" not in st.session_state:
    st.session_state.daily_summaries = {}
if "rag_index" not in st.session_state:
    st.session_state.rag_index = {}
if "last_updated" not in st.session_state:
    st.session_state.last_updated = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "news_data" not in st.session_state:
    st.session_state.news_data = {}
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Define ticker mapping by category
TICKERS = {
    "crypto": {
        "BTC": {"name": "Bitcoin", "alpha_symbol": "CRYPTO:BTC"},
        "ETH": {"name": "Ethereum", "alpha_symbol": "CRYPTO:ETH"},
        "BNB": {"name": "Binance Coin", "alpha_symbol": "CRYPTO:BNB"},
        "SOL": {"name": "Solana", "alpha_symbol": "CRYPTO:SOL"},
        "ADA": {"name": "Cardano", "alpha_symbol": "CRYPTO:ADA"},
        "DOT": {"name": "Polkadot", "alpha_symbol": "CRYPTO:DOT"},
        "XRP": {"name": "Ripple", "alpha_symbol": "CRYPTO:XRP"},
        "DOGE": {"name": "Dogecoin", "alpha_symbol": "CRYPTO:DOGE"},
        "AVAX": {"name": "Avalanche", "alpha_symbol": "CRYPTO:AVAX"},
        "LINK": {"name": "Chainlink", "alpha_symbol": "CRYPTO:LINK"}
    },
    "stock": {
        "AAPL": {"name": "Apple Inc.", "alpha_symbol": "AAPL"},
        "MSFT": {"name": "Microsoft", "alpha_symbol": "MSFT"},
        "GOOGL": {"name": "Alphabet", "alpha_symbol": "GOOGL"},
        "AMZN": {"name": "Amazon", "alpha_symbol": "AMZN"},
        "META": {"name": "Meta Platforms", "alpha_symbol": "META"},
        "TSLA": {"name": "Tesla", "alpha_symbol": "TSLA"},
        "NVDA": {"name": "NVIDIA", "alpha_symbol": "NVDA"},
        "JPM": {"name": "JPMorgan Chase", "alpha_symbol": "JPM"},
        "V": {"name": "Visa", "alpha_symbol": "V"},
        "WMT": {"name": "Walmart", "alpha_symbol": "WMT"}
    },
    "index": {
        "SPX": {"name": "S&P 500", "alpha_symbol": "INDEX:SPX"},
        "DJI": {"name": "Dow Jones", "alpha_symbol": "INDEX:DJI"},
        "COMP": {"name": "NASDAQ", "alpha_symbol": "INDEX:COMP"},
        "RUT": {"name": "Russell 2000", "alpha_symbol": "INDEX:RUT"},
        "VIX": {"name": "Volatility Index", "alpha_symbol": "INDEX:VIX"}
    },
    "forex": {
        "EUR/USD": {"name": "Euro/US Dollar", "alpha_symbol": "FOREX:EUR/USD"},
        "USD/JPY": {"name": "US Dollar/Japanese Yen", "alpha_symbol": "FOREX:USD/JPY"},
        "GBP/USD": {"name": "British Pound/US Dollar", "alpha_symbol": "FOREX:GBP/USD"},
        "USD/CHF": {"name": "US Dollar/Swiss Franc", "alpha_symbol": "FOREX:USD/CHF"},
        "AUD/USD": {"name": "Australian Dollar/US Dollar", "alpha_symbol": "FOREX:AUD/USD"}
    },
    "commodity": {
        "GOLD": {"name": "Gold", "alpha_symbol": "COMMODITY:GOLD"},
        "SILVER": {"name": "Silver", "alpha_symbol": "COMMODITY:SILVER"},
        "OIL": {"name": "Crude Oil", "alpha_symbol": "COMMODITY:OIL"},
        "NATGAS": {"name": "Natural Gas", "alpha_symbol": "COMMODITY:NATGAS"},
        "COPPER": {"name": "Copper", "alpha_symbol": "COMMODITY:COPPER"}
    }
}

# Configure data directory
DATA_DIR = "financial_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Function to fetch news from Alpha Vantage
def fetch_alpha_vantage_news(asset_category, asset_symbol, start_date=None, end_date=None):
    """Fetch news related to the ticker from Alpha Vantage API"""
    # Determine search topics based on asset category
    if asset_category == "crypto":
        topics = "blockchain,cryptocurrency"
    elif asset_category == "commodity":
        topics = "economy_fiscal,economy_monetary"
    elif asset_category in ["stock", "index"]:
        topics = "earnings,ipo,mergers_and_acquisitions,financial_markets"
    elif asset_category == "forex":
        topics = "forex,economy_fiscal,economy_monetary"
    else:
        topics = "financial_markets"
    
    # Determine tickers to search
    alpha_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"]
    if '/' in alpha_symbol:  # Handle forex symbols with slashes
        alpha_symbol = alpha_symbol.split(':')[-1]
    
    # Set time range for news
    if not start_date:
        time_from = (datetime.now() - timedelta(days=7)).strftime("%Y%m%dT0000")
    else:
        time_from = start_date.strftime("%Y%m%dT0000")
    
    if end_date:
        time_to = end_date.strftime("%Y%m%dT2359")
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topics,
            "tickers": alpha_symbol,
            "time_from": time_from,
            "time_to": time_to,
            "limit": 50,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    else:
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topics,
            "tickers": alpha_symbol,
            "time_from": time_from,
            "limit": 50,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        data = response.json()
        
        if "feed" not in data:
            st.warning(f"No news data returned from Alpha Vantage API. Response: {data}")
            return []
            
        return data["feed"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news from Alpha Vantage: {e}")
        return []

# Function to fetch market data from Alpha Vantage
def fetch_alpha_vantage_data(asset_category, asset_symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    """Fetch market data from Alpha Vantage API"""
    alpha_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"]
    if ":" in alpha_symbol:
        alpha_symbol = alpha_symbol.split(":")[-1]
    
    # Select appropriate function based on asset category
    if asset_category == "crypto":
        function = "DIGITAL_CURRENCY_DAILY"
        params = {
            "function": function,
            "symbol": alpha_symbol,
            "market": "USD",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    elif asset_category == "forex":
        function = "FX_DAILY"
        from_symbol, to_symbol = alpha_symbol.split("/")
        params = {
            "function": function,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    else:  # Stocks, ETFs, indices
        params = {
            "function": function,
            "symbol": alpha_symbol,
            "outputsize": outputsize,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"Alpha Vantage API error: {data['Error Message']}")
            return pd.DataFrame()
            
        # Parse response based on function type
        if function == "DIGITAL_CURRENCY_DAILY":
            time_series_key = "Time Series (Digital Currency Daily)"
            if time_series_key not in data:
                st.error("Expected data format not found in Alpha Vantage response")
                return pd.DataFrame()
                
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                f"1a. open (USD)": "Open",
                f"2a. high (USD)": "High",
                f"3a. low (USD)": "Low",
                f"4a. close (USD)": "Close",
                f"5. volume": "Volume"
            })
        elif function == "FX_DAILY":
            time_series_key = "Time Series FX (Daily)"
            if time_series_key not in data:
                st.error("Expected data format not found in Alpha Vantage response")
                return pd.DataFrame()
                
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close"
            })
        else:  # TIME_SERIES_DAILY
            possible_keys = ["Time Series (Daily)", "Weekly Time Series", "Monthly Time Series"]
            time_series_key = next((key for key in possible_keys if key in data), None)
                
            if not time_series_key:
                st.error("Expected time series data not found in Alpha Vantage response")
                return pd.DataFrame()
                
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df.sort_index()
        
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()

# Function to create RAG system
def create_rag_system(asset_category, asset_symbol, news_articles):
    """Create a RAG system from the fetched news articles"""
    if not news_articles:
        return None
    
    temp_dir = os.path.join(DATA_DIR, f"{asset_category}_{asset_symbol}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save articles to text files
    article_files = []
    for i, article in enumerate(news_articles):
        title = article.get("title", "No title")
        summary = article.get("summary", "No content")
        url = article.get("url", "No URL")
        time_published = article.get("time_published", "No date")
        authors = article.get("authors", [])
        source = article.get("source", "Unknown source")
        
        # Get sentiment data if available
        overall_sentiment_score = article.get("overall_sentiment_score", "N/A")
        overall_sentiment_label = article.get("overall_sentiment_label", "N/A")
        ticker_sentiment = "N/A"
        
        if "ticker_sentiment" in article:
            ticker_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]
            for sentiment in article["ticker_sentiment"]:
                if sentiment["ticker"] == ticker_symbol:
                    ticker_sentiment = f"Score: {sentiment.get('ticker_sentiment_score', 'N/A')}, Label: {sentiment.get('ticker_sentiment_label', 'N/A')}"
                    break
        
        # Combine all information for the document
        content = f"""TITLE: {title}
DATE: {time_published}
SOURCE: {source}
AUTHORS: {', '.join(authors) if authors else 'Unknown'}
URL: {url}
OVERALL SENTIMENT SCORE: {overall_sentiment_score}
OVERALL SENTIMENT LABEL: {overall_sentiment_label}
TICKER SPECIFIC SENTIMENT ({TICKERS[asset_category][asset_symbol]['name']}): {ticker_sentiment}

SUMMARY:
{summary}
"""
        
        file_path = os.path.join(temp_dir, f"article_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        article_files.append(file_path)
    
    # Create a text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Load and split documents
    documents = []
    for file_path in article_files:
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()
        docs = text_splitter.split_documents(loaded_docs)
        documents.extend(docs)
    
    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector database
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# Function to call OpenRouter API for LLM tasks
def openrouter_llm_call(prompt, max_tokens=1000):
    """Call the OpenRouter API for LLM tasks"""
    try:
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://financial-insights-hub.streamlit.app",
                "X-Title": "Financial Insights Hub",
            },
            model="nvidia/llama-3.1-nemotron-70b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenRouter API: {e}")
        return f"Error generating response: {str(e)}"

# Function to summarize articles
def summarize_daily_news(asset_category, asset_symbol, news_articles, date_str):
    """Generate a summary for the ticker for a specific date using OpenRouter"""
    
    # Filter articles for the specified date
    date_articles = [
        article for article in news_articles 
        if date_str in article.get("time_published", "")
    ]
    
    if not date_articles:
        return "No articles found for this date."
    
    # Prepare content for summarization
    content = ""
    for article in date_articles[:5]:  # Limit to 5 articles to avoid token limits
        title = article.get("title", "No title")
        summary = article.get("summary", "")
        source = article.get("source", "Unknown")
        
        # Get sentiment if available
        sentiment_info = ""
        if "overall_sentiment_label" in article:
            sentiment_info = f"Overall sentiment: {article['overall_sentiment_label']}"
            
            # Check if there's ticker-specific sentiment
            if "ticker_sentiment" in article:
                ticker_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]
                for sentiment in article["ticker_sentiment"]:
                    if sentiment.get("ticker") == ticker_symbol:
                        sentiment_info += f", {TICKERS[asset_category][asset_symbol]['name']} sentiment: {sentiment.get('ticker_sentiment_label', 'N/A')}"
                        break
        
        content += f"TITLE: {title}\nSOURCE: {source}\n{sentiment_info}\n\nSUMMARY: {summary}\n\n---\n\n"
    
    # Create prompt template for summarization
    template = """
    You are a financial analyst specializing in market analysis. Below are news articles about {asset_name} from {date}.
    
    {content}
    
    Please provide:
    1. A concise summary of the key developments regarding {asset_name} on this date
    2. The overall sentiment (positive, negative, or neutral)
    3. Any potential market impact mentioned
    4. Key factors driving the market sentiment
    
    Keep your analysis focused on factual information from the articles.
    """
    
    # Format the prompt
    formatted_prompt = template.format(
        asset_name=TICKERS[asset_category][asset_symbol]["name"],
        date=date_str,
        content=content
    )
    
    # Call OpenRouter API for the summary
    summary = openrouter_llm_call(formatted_prompt, max_tokens=1000)
    
    return summary

# Function to calculate sentiment score
def calculate_sentiment_score(asset_category, asset_symbol, news_articles):
    """Calculate sentiment score from Alpha Vantage news articles"""
    if not news_articles:
        return 0
        
    total_score = 0
    count = 0
    
    # First try ticker-specific sentiment
    ticker_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]
    
    for article in news_articles:
        if "ticker_sentiment" not in article:
            continue
            
        for sentiment in article["ticker_sentiment"]:
            if sentiment.get("ticker") == ticker_symbol:
                if "ticker_sentiment_score" in sentiment:
                    total_score += float(sentiment["ticker_sentiment_score"])
                    count += 1
    
    # If no ticker-specific sentiment, use overall sentiment
    if count == 0:
        for article in news_articles:
            if "overall_sentiment_score" in article:
                total_score += float(article["overall_sentiment_score"])
                count += 1
    
    # Calculate average
    return total_score / count if count > 0 else 0

# Function to ask questions to the RAG system
def ask_rag(question, vectorstore, asset_category, asset_symbol):
    """Ask a question to the RAG system"""
    if not vectorstore:
        return "No data available for this asset. Please try another asset or refresh the data."
    
    # Get relevant documents from the RAG system
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    
    # Extract context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt template for Q&A
    template = """
    You are an AI assistant specializing in financial analysis for {asset_name}.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer or the information isn't in the context, just say so - don't make up information.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    
    # Format the prompt
    formatted_prompt = template.format(
        asset_name=TICKERS[asset_category][asset_symbol]["name"],
        context=context,
        question=question
    )
    
    # Call OpenRouter API for the answer
    answer = openrouter_llm_call(formatted_prompt, max_tokens=1000)
    
    return answer

# Function to predict tomorrow's price
def predict_price(asset_category, asset_symbol, data, sentiment_score):
    """Predict tomorrow's price based on technical indicators and sentiment"""
    if data.empty or len(data) < 50:
        return "Insufficient data for price prediction"
    
    # Calculate key technical indicators
    df = data.copy()
    
    # Add technical indicators using pandas_ta
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Prepare prompt for price prediction
    last_close = latest['Close']
    rsi = latest.get('RSI_14', np.nan)
    macd = latest.get('MACD_12_26_9', np.nan)
    macd_signal = latest.get('MACDs_12_26_9', np.nan)
    bb_upper = latest.get('BBU_20_2.0', np.nan)
    bb_lower = latest.get('BBL_20_2.0', np.nan)
    
    # Calculate short-term trend (last 5 days)
    short_term_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
    
    template = f"""
    As a financial analyst, predict tomorrow's price for {TICKERS[asset_category][asset_symbol]['name']} based on the following data:
    
    Current price: ${last_close:.2f}
    5-day change: {short_term_change:.2f}%
    RSI (14): {rsi:.2f}
    MACD: {macd:.4f}
    MACD Signal: {macd_signal:.4f}
    Bollinger Band Upper: ${bb_upper:.2f}
    Bollinger Band Lower: ${bb_lower:.2f}
    News Sentiment Score (scale -1 to 1): {sentiment_score:.2f}
    
    Based on technical analysis and market sentiment, provide:
    1. A specific price prediction for tomorrow
    2. The reasoning behind this prediction
    3. Confidence level (low, medium, high)
    4. Key factors influencing your prediction
    
    Be concise but specific with numbers.
    """
    
    # Call LLM for prediction
    prediction = openrouter_llm_call(template, max_tokens=800)
    return prediction

# Function to display the home page
def display_home_page():
    st.title("🌟 Financial Insights Hub")
    
    st.markdown("""
    ### Welcome to your comprehensive financial analysis platform!
    
    This application provides real-time financial insights, news analysis, and technical indicators for various asset classes.
    Powered by Alpha Vantage data and advanced AI analysis, it helps you make informed financial decisions.
    
    #### 🔑 Key Features:
    
    - **Multi-Asset Coverage**: Analyze cryptocurrencies, stocks, indices, forex, and commodities
    - **News Sentiment Analysis**: Track market sentiment from financial news
    - **Technical Indicators**: View key technical analysis metrics using Pandas TA
    - **AI-Powered Insights**: Ask questions about market trends and get AI-analyzed responses
    - **Price Predictions**: AI-generated price forecasts based on technical and sentiment data
    - **Custom Date Ranges**: Select specific timeframes for your analysis
    
    #### 📊 Getting Started:
    
    1. Choose an asset category and specific asset from the sidebar
    2. Select your desired date range for news analysis
    3. Explore different views: Overview, News Analysis, or Technical Analysis
    4. Use the AI assistant to ask specific questions about your selected asset
    
    Use the sidebar to navigate through different assets and refresh data when needed.
    """)
    
    # Show a sample visualization
    st.subheader("Market Sentiment Overview")
    
    # Create sample data for visualization
    categories = ["Crypto", "Stocks", "Forex", "Commodities", "Indices"]
    sentiments = [0.45, 0.12, -0.25, 0.32, -0.15]  # Sample sentiment scores
    
    fig = px.bar(
        x=categories,
        y=sentiments,
        color=sentiments,
        color_continuous_scale=["red", "gray", "green"],
        range_color=[-1, 1],
        labels={"x": "Asset Category", "y": "Market Sentiment"},
        title="Current Market Sentiment by Asset Class"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This is a conceptual visualization. Select an asset from the sidebar to see real-time data analysis.")

# Function to display overview 
def display_overview(asset_category, asset_symbol, news_articles, data):
    """Display overview information for the selected asset"""
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        st.subheader("Price Chart")
        
        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            fig.update_layout(
                title=f"{TICKERS[asset_category][asset_symbol]['name']} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No financial data available from Alpha Vantage.")
    
    with col2:
        # Current price and stats
        if not data.empty:
            latest_data = data.iloc[-1]
            prev_data = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
            
            # Calculate change
            live_price = latest_data["Close"]
            prev_close = prev_data["Close"]
            change = live_price - prev_close
            change_pct = (change / prev_close) * 100
            
            st.metric(
                label="Current Price", 
                value=f"${live_price:.2f}", 
                delta=f"{change:.2f} ({change_pct:.2f}%)"
            )
            
            # Display volume if available
            if "Volume" in data.columns:
                volume = latest_data["Volume"]
                if not np.isnan(volume):
                    if volume >= 1_000_000_000:
                        volume_str = f"{volume/1_000_000_000:.2f}B"
                    elif volume >= 1_000_000:
                        volume_str = f"{volume/1_000_000:.2f}M"
                    else:
                        volume_str = f"{volume:,.0f}"
                    
                    st.metric(label="Volume", value=volume_str)
            
            # 52-week range
            try:
                year_data = data.last('365D')
                week_low = year_data['Low'].min()
                week_high = year_data['High'].max()
                st.metric(label="52-Week Range", value=f"${week_low:.2f} - ${week_high:.2f}")
            except:
                pass
                
        else:
            st.warning("No price data available to display metrics.")
            
        # News sentiment overview
        st.subheader("News Sentiment")
        
        if news_articles:
            # Calculate sentiment from Alpha Vantage news
            avg_sentiment = calculate_sentiment_score(asset_category, asset_symbol, news_articles)
            
            # Create gauge chart for sentiment
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "News Sentiment"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.33], 'color': "red"},
                        {'range': [-0.33, 0.33], 'color': "gray"},
                        {'range': [0.33, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_sentiment
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent headlines with sentiment
            st.subheader("Recent Headlines")
            
            for article in news_articles[:3]:  # Show top 3 recent headlines
                title = article.get("title", "No title")
                sentiment = "N/A"
                
                # Try to get ticker-specific sentiment
                if "ticker_sentiment" in article:
                    ticker_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]
                    for ts in article["ticker_sentiment"]:
                        if ts.get("ticker") == ticker_symbol:
                            sentiment = ts.get("ticker_sentiment_label", "N/A")
                            break
                
                # If no ticker-specific sentiment, use overall
                if sentiment == "N/A" and "overall_sentiment_label" in article:
                    sentiment = article["overall_sentiment_label"]
                
                # Display headline with sentiment color
                if sentiment.lower() == "bullish" or sentiment.lower() == "positive":
                    st.markdown(f"🟢 **{title}** - *{sentiment}*")
                elif sentiment.lower() == "bearish" or sentiment.lower() == "negative":
                    st.markdown(f"🔴 **{title}** - *{sentiment}*")
                else:
                    st.markdown(f"⚪ **{title}** - *{sentiment}*")
        else:
            st.warning("No news articles available for sentiment analysis.")

    # Show AI prediction
    st.subheader("AI Price Prediction")
    
    if asset_symbol in st.session_state.predictions:
        st.markdown(st.session_state.predictions[asset_symbol])
    else:
        if not data.empty and news_articles:
            sentiment_score = calculate_sentiment_score(asset_category, asset_symbol, news_articles)
            prediction = predict_price(asset_category, asset_symbol, data, sentiment_score)
            st.session_state.predictions[asset_symbol] = prediction
            st.markdown(prediction)
        else:
            st.warning("Insufficient data for price prediction. Please refresh data.")

# Function to display news analysis
def display_news_analysis(asset_category, asset_symbol, news_articles, start_date, end_date):
    """Display news analysis for the selected asset"""
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} News Analysis")
    
    if not news_articles:
        st.warning("No news articles available for the selected date range.")
        return
    
    # Organize news by date
    news_by_date = {}
    for article in news_articles:
        date_str = article.get("time_published", "")[:10]  # Get YYYY-MM-DD part
        if date_str:
            if date_str not in news_by_date:
                news_by_date[date_str] = []
            news_by_date[date_str].append(article)
    
    # Sort dates
    dates = sorted(news_by_date.keys(), reverse=True)
    
    # Display news count by date
    st.subheader("News Volume by Date")
    
    date_counts = {date: len(articles) for date, articles in news_by_date.items()}
    
    fig = px.bar(
        x=list(date_counts.keys()),
        y=list(date_counts.values()),
        labels={"x": "Date", "y": "Number of Articles"},
        title="News Volume"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for Daily Summaries and All Articles
    tab1, tab2, tab3 = st.tabs(["📝 Daily Summaries", "📰 All Articles", "❓ Ask about News"])
    
    with tab1:
        # Show AI-generated daily summaries
        st.subheader("AI-Generated Daily News Summaries")
        
        for date in dates:
            with st.expander(f"Summary for {date}"):
                # Check if summary exists in session state
                key = f"{asset_symbol}_{date}"
                if key in st.session_state.daily_summaries:
                    st.markdown(st.session_state.daily_summaries[key])
                else:
                    # Generate new summary
                    with st.spinner(f"Generating summary for {date}..."):
                        summary = summarize_daily_news(
                            asset_category, 
                            asset_symbol, 
                            news_by_date[date], 
                            date
                        )
                        st.session_state.daily_summaries[key] = summary
                        st.markdown(summary)
    
    with tab2:
        # Show all articles in a table
        st.subheader("All News Articles")
        
        # Create a dataframe for displaying articles
        articles_data = []
        for article in news_articles:
            # Extract basic article info
            title = article.get("title", "No title")
            summary = article.get("summary", "No summary")
            date = article.get("time_published", "")[:10]
            source = article.get("source", "Unknown")
            url = article.get("url", "#")
            
            # Get sentiment
            sentiment = "N/A"
            sentiment_score = 0.0
            
            # Try to get ticker-specific sentiment
            if "ticker_sentiment" in article:
                ticker_symbol = TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]
                for ts in article["ticker_sentiment"]:
                    if ts.get("ticker") == ticker_symbol:
                        sentiment = ts.get("ticker_sentiment_label", "N/A")
                        sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                        break
            
            # If no ticker-specific sentiment, use overall
            if sentiment == "N/A" and "overall_sentiment_label" in article:
                sentiment = article["overall_sentiment_label"]
                sentiment_score = float(article.get("overall_sentiment_score", 0))
            
            articles_data.append({
                "Date": date,
                "Source": source,
                "Title": title,
                "Sentiment": sentiment,
                "Sentiment Score": sentiment_score,
                "URL": url
            })
        
        # Create dataframe and display
        if articles_data:
            df = pd.DataFrame(articles_data)
            
            # Add color based on sentiment
            def color_sentiment(val):
                if 'bullish' in val.lower() or 'positive' in val.lower():
                    return 'background-color: rgba(0, 128, 0, 0.2)'
                elif 'bearish' in val.lower() or 'negative' in val.lower():
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                else:
                    return ''
            
            # Display the styled dataframe
            st.dataframe(
                df.style.applymap(color_sentiment, subset=['Sentiment']),
                column_config={
                    "URL": st.column_config.LinkColumn("Link")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No articles available to display.")
    
    with tab3:
        # QA System for asking questions about the news
        st.subheader("Ask Questions About the News")
        
        # Create RAG system if not already in session state
        key = f"{asset_category}_{asset_symbol}"
        if key not in st.session_state.rag_index:
            with st.spinner("Building knowledge base from news articles..."):
                vectorstore = create_rag_system(asset_category, asset_symbol, news_articles)
                st.session_state.rag_index[key] = vectorstore
        
        # Input for user question
        user_question = st.text_input("Ask a question about the news for this asset:", key=f"question_{key}")
        
        if user_question:
            # Initialize chat history if needed
            if key not in st.session_state.chat_history:
                st.session_state.chat_history[key] = []
            
            # Store question in history
            st.session_state.chat_history[key].append({"role": "user", "content": user_question})
            
            # Get answer from RAG system
            with st.spinner("Analyzing news articles to answer your question..."):
                answer = ask_rag(user_question, st.session_state.rag_index[key], asset_category, asset_symbol)
                
                # Store answer in history
                st.session_state.chat_history[key].append({"role": "assistant", "content": answer})
        
        # Display chat history
        if key in st.session_state.chat_history:
            st.subheader("Conversation History")
            for message in st.session_state.chat_history[key]:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("---")

# Function to display technical analysis
def display_technical_analysis(asset_category, asset_symbol, data):
    """Display technical analysis for the selected asset"""
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} Technical Analysis")
    
    if data.empty:
        st.error("No financial data available for technical analysis.")
        return
    
    # Calculate technical indicators
    df = data.copy()
    
    # Add Pandas TA indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3 = st.tabs(["📊 Price & Moving Averages", "🔍 Oscillators", "📈 Advanced Analysis"])
    
    with tab1:
        # Price chart with moving averages
        st.subheader("Price with Moving Averages")
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['SMA_20'], 
            line=dict(color='blue', width=1),
            name='SMA 20'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['SMA_50'], 
            line=dict(color='orange', width=1),
            name='SMA 50'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['SMA_200'], 
            line=dict(color='red', width=1),
            name='SMA 200'
        ))
        
        fig.update_layout(
            title=f"{TICKERS[asset_category][asset_symbol]['name']} Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current moving average values
        st.subheader("Moving Average Analysis")
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        sma20 = latest['SMA_20']
        sma50 = latest['SMA_50']
        sma200 = latest['SMA_200']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="SMA 20", 
                value=f"${sma20:.2f}", 
                delta=f"{((current_price/sma20)-1)*100:.2f}%" if not np.isnan(sma20) else None
            )
        
        with col2:
            st.metric(
                label="SMA 50", 
                value=f"${sma50:.2f}", 
                delta=f"{((current_price/sma50)-1)*100:.2f}%" if not np.isnan(sma50) else None
            )
        
        with col3:
            st.metric(
                label="SMA 200", 
                value=f"${sma200:.2f}", 
                delta=f"{((current_price/sma200)-1)*100:.2f}%" if not np.isnan(sma200) else None
            )
        
        # Moving average crossover analysis
        st.subheader("Moving Average Crossover Analysis")
        
        # Golden Cross / Death Cross
        if not np.isnan(sma50) and not np.isnan(sma200):
            if sma50 > sma200:
                # Check for recent golden cross
                previous_10_days = df.iloc[-10:].copy()
                crossover = False
                for i in range(1, len(previous_10_days)):
                    if (previous_10_days['SMA_50'].iloc[i-1] <= previous_10_days['SMA_200'].iloc[i-1] and 
                        previous_10_days['SMA_50'].iloc[i] > previous_10_days['SMA_200'].iloc[i]):
                        crossover = True
                        crossover_date = previous_10_days.index[i]
                        break
                
                if crossover:
                    st.success(f"🚀 **Golden Cross detected on {crossover_date.date()}!** The 50-day SMA crossed above the 200-day SMA, a bullish signal.")
                else:
                    st.info("📈 **Bullish Trend**: The 50-day SMA is above the 200-day SMA.")
            else:
                # Check for recent death cross
                previous_10_days = df.iloc[-10:].copy()
                crossover = False
                for i in range(1, len(previous_10_days)):
                    if (previous_10_days['SMA_50'].iloc[i-1] >= previous_10_days['SMA_200'].iloc[i-1] and 
                        previous_10_days['SMA_50'].iloc[i] < previous_10_days['SMA_200'].iloc[i]):
                        crossover = True
                        crossover_date = previous_10_days.index[i]
                        break
                        
                if crossover:
                    st.error(f"⚠️ **Death Cross detected on {crossover_date.date()}!** The 50-day SMA crossed below the 200-day SMA, a bearish signal.")
                else:
                    st.warning("📉 **Bearish Trend**: The 50-day SMA is below the 200-day SMA.")
    
    with tab2:
        # RSI Chart
        st.subheader("Relative Strength Index (RSI)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['RSI_14'], 
            line=dict(color='purple', width=2),
            name='RSI (14)'
        ))
        
        # Add horizontal lines at 70 and 30
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="RSI (14) - Relative Strength Index",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI interpretation
        latest_rsi = df['RSI_14'].iloc[-1]
        if not np.isnan(latest_rsi):
            st.metric("Current RSI (14)", f"{latest_rsi:.2f}")
            
            if latest_rsi > 70:
                st.warning("🔥 **Overbought**: RSI above 70 indicates the asset may be overbought and could be due for a pullback.")
            elif latest_rsi < 30:
                st.success("❄️ **Oversold**: RSI below 30 indicates the asset may be oversold and could be due for a rebound.")
            else:
                st.info("🔄 **Neutral RSI Zone**: The asset is neither overbought nor oversold according to RSI.")
        
        # MACD Chart
        st.subheader("MACD - Moving Average Convergence Divergence")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # Add price to top subplot
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add MACD to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                line=dict(color='blue', width=1.5),
                name='MACD'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACDs_12_26_9'],
                line=dict(color='red', width=1.5),
                name='Signal'
            ),
            row=2, col=1
        )
        
        # Add histogram for MACD
        macd_hist = df['MACDh_12_26_9']
        colors = ['green' if val >= 0 else 'red' for val in macd_hist]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=macd_hist,
                marker_color=colors,
                name='Histogram'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="MACD (12,26,9)",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MACD interpretation
        latest_macd = df['MACD_12_26_9'].iloc[-1]
        latest_signal = df['MACDs_12_26_9'].iloc[-1]
        latest_hist = df['MACDh_12_26_9'].iloc[-1]
        
        if not np.isnan(latest_macd) and not np.isnan(latest_signal):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MACD", f"{latest_macd:.4f}")
            with col2:
                st.metric("Signal", f"{latest_signal:.4f}")
            with col3:
                st.metric("Histogram", f"{latest_hist:.4f}")
            
            # MACD interpretation
            if latest_macd > latest_signal:
                if latest_macd > 0:
                    st.success("🚀 **Strong Bullish Signal**: MACD is above the signal line and positive, indicating strong bullish momentum.")
                else:
                    st.info("📈 **Bullish Signal**: MACD is above the signal line but still negative, indicating potential bullish momentum building.")
            else:
                if latest_macd < 0:
                    st.error("📉 **Strong Bearish Signal**: MACD is below the signal line and negative, indicating strong bearish momentum.")
                else:
                    st.warning("⚠️ **Bearish Signal**: MACD is below the signal line but still positive, indicating potential bearish momentum building.")
    
    with tab3:
        # Bollinger Bands
        st.subheader("Bollinger Bands")
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['BBU_20_2.0'], 
            line=dict(color='gray', width=1),
            name='Upper BB'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['BBM_20_2.0'], 
            line=dict(color='blue', width=1),
            name='Middle BB'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['BBL_20_2.0'], 
            line=dict(color='gray', width=1),
            name='Lower BB',
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.update_layout(
            title="Bollinger Bands (20,2)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bollinger Bands interpretation
        latest_price = df['Close'].iloc[-1]
        latest_bbu = df['BBU_20_2.0'].iloc[-1]
        latest_bbm = df['BBM_20_2.0'].iloc[-1]
        latest_bbl = df['BBL_20_2.0'].iloc[-1]
        
        if not np.isnan(latest_bbu) and not np.isnan(latest_bbl):
            # Calculate bandwidth and %B
            bandwidth = (latest_bbu - latest_bbl) / latest_bbm * 100
            percent_b = (latest_price - latest_bbl) / (latest_bbu - latest_bbl) * 100 if (latest_bbu - latest_bbl) != 0 else 50
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bandwidth", f"{bandwidth:.2f}%")
            with col2:
                st.metric("%B", f"{percent_b:.2f}%")
            
            # Interpretation
            if latest_price > latest_bbu:
                st.warning("🚨 **Overbought**: Price is above the upper Bollinger Band, suggesting the asset may be overbought.")
            elif latest_price < latest_bbl:
                st.success("⭐ **Oversold**: Price is below the lower Bollinger Band, suggesting the asset may be oversold.")
            else:
                st.info("➡️ **Within Bands**: Price is within the Bollinger Bands, indicating normal volatility.")
            
            # Bandwidth interpretation
            if bandwidth < 10:
                st.info("📊 **Low Volatility**: The Bollinger Bands are narrowing, which often precedes a period of higher volatility.")
            elif bandwidth > 30:
                st.warning("📊 **High Volatility**: The Bollinger Bands are wide, indicating high market volatility.")

# Main function for the Streamlit app
def main():
    # Sidebar for navigation
    st.sidebar.title("Financial Insights Hub")
    
    # Create sidebar navigation
    selected_page = st.sidebar.radio(
        "Navigation",
        ["Home", "Asset Analysis"]
    )
    
    if selected_page == "Home":
        display_home_page()
        
    elif selected_page == "Asset Analysis":
        # Asset selection in sidebar
        st.sidebar.header("Asset Selection")
        
        # Select asset category
        asset_category = st.sidebar.selectbox(
            "Asset Category",
            list(TICKERS.keys()),
            format_func=lambda x: x.capitalize()
        )
        
        # Select specific asset
        asset_symbol = st.sidebar.selectbox(
            f"Select {asset_category.capitalize()}",
            list(TICKERS[asset_category].keys()),
            format_func=lambda x: f"{x} - {TICKERS[asset_category][x]['name']}"
        )
        
        # Date range selection
        st.sidebar.header("Date Range")
        
        # Add date inputs
        today = datetime.now().date()
        start_date = st.sidebar.date_input(
            "Start Date",
            value=today - timedelta(days=30),
            max_value=today
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=today,
            max_value=today,
            min_value=start_date
        )
        
        # Refresh button
        refresh_data = st.sidebar.button("Refresh Data")
        
        # Fetch data
        data_key = f"{asset_category}_{asset_symbol}"
        news_key = f"{asset_category}_{asset_symbol}_news"
        
        # Check if we need to fetch new data
        if refresh_data or data_key not in st.session_state.last_updated:
            with st.spinner("Fetching financial data..."):
                data = fetch_alpha_vantage_data(asset_category, asset_symbol)
                st.session_state[data_key] = data
                st.session_state.last_updated[data_key] = datetime.now()
        else:
            data = st.session_state.get(data_key, pd.DataFrame())
        
        # Check if we need to fetch new news data
        if refresh_data or news_key not in st.session_state.last_updated:
            with st.spinner("Fetching news data..."):
                news_articles = fetch_alpha_vantage_news(
                    asset_category, 
                    asset_symbol,
                    start_date,
                    end_date
                )
                st.session_state[news_key] = news_articles
                st.session_state.last_updated[news_key] = datetime.now()
                st.session_state.news_data[news_key] = news_articles
        else:
            news_articles = st.session_state.get(news_key, [])
        
        # Display last updated time
        if data_key in st.session_state.last_updated:
            last_updated = st.session_state.last_updated[data_key]
            st.sidebar.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Overview", "News Analysis", "Technical Analysis"])
        
        with tab1:
            display_overview(asset_category, asset_symbol, news_articles, data)
        
        with tab2:
            display_news_analysis(asset_category, asset_symbol, news_articles, start_date, end_date)
        
        with tab3:
            display_technical_analysis(asset_category, asset_symbol, data)

# Entry point
if __name__ == "__main__":
    main()