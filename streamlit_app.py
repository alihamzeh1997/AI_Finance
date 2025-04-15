# app.py - Main Streamlit Application with Alpha Vantage
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import os
import time
import json
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from openai import OpenAI
import tempfile
import plotly.graph_objs as go
import plotly.express as px

# Configuration and API Keys
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]  # Set this in Streamlit secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]  # Set this in Streamlit secrets

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

# Initialize session state variables if they don't exist
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

# Define ticker mapping (symbol to readable name and Yahoo Finance symbol)
TICKERS = {
    "bitcoin": {"name": "Bitcoin (BTC)", "symbol": "BTC-USD", "type": "crypto", "alpha_symbol": "CRYPTO:BTC"},
    "ethereum": {"name": "Ethereum (ETH)", "symbol": "ETH-USD", "type": "crypto", "alpha_symbol": "CRYPTO:ETH"},
    "gold": {"name": "Gold", "symbol": "GC=F", "type": "commodity", "alpha_symbol": "COMMODITY:GOLD"},
    "silver": {"name": "Silver", "symbol": "SI=F", "type": "commodity", "alpha_symbol": "COMMODITY:SILVER"},
    "sp500": {"name": "S&P 500", "symbol": "^GSPC", "type": "index", "alpha_symbol": "INDEX:SPX"},
    "dow": {"name": "Dow Jones", "symbol": "^DJI", "type": "index", "alpha_symbol": "INDEX:DJI"},
    "nasdaq": {"name": "NASDAQ", "symbol": "^IXIC", "type": "index", "alpha_symbol": "INDEX:COMP"},
    "apple": {"name": "Apple Inc.", "symbol": "AAPL", "type": "stock", "alpha_symbol": "AAPL"},
    "microsoft": {"name": "Microsoft", "symbol": "MSFT", "type": "stock", "alpha_symbol": "MSFT"},
    "tesla": {"name": "Tesla", "symbol": "TSLA", "type": "stock", "alpha_symbol": "TSLA"},
    "amazon": {"name": "Amazon", "symbol": "AMZN", "type": "stock", "alpha_symbol": "AMZN"},
    "eurusd": {"name": "EUR/USD", "symbol": "EURUSD=X", "type": "forex", "alpha_symbol": "FOREX:EUR/USD"},
    "usdjpy": {"name": "USD/JPY", "symbol": "JPY=X", "type": "forex", "alpha_symbol": "FOREX:USD/JPY"},
}

# Configure data directory for storing article data
DATA_DIR = "financial_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Function to fetch news from Alpha Vantage
def fetch_alpha_vantage_news(ticker, days_back=7):
    """Fetch news related to the ticker from Alpha Vantage API"""
    # Determine search topics based on ticker type
    if TICKERS[ticker]["type"] == "crypto":
        topics = "blockchain,cryptocurrency"
    elif TICKERS[ticker]["type"] == "commodity":
        topics = "economy_fiscal,economy_monetary"
    elif TICKERS[ticker]["type"] in ["stock", "index"]:
        topics = "earnings,ipo,mergers_and_acquisitions,financial_markets"
    elif TICKERS[ticker]["type"] == "forex":
        topics = "forex,economy_fiscal,economy_monetary"
    else:
        topics = "financial_markets"
    
    # Determine tickers to search
    alpha_symbol = TICKERS[ticker]["alpha_symbol"]
    if '/' in alpha_symbol:  # Handle forex symbols with slashes
        alpha_symbol = alpha_symbol.split(':')[-1]
    
    # Make API call
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "tickers": alpha_symbol,
        "time_from": (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%dT0000"),
        "limit": 50,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        data = response.json()
        
        # Check if we have valid data
        if "feed" not in data:
            st.warning(f"No news data returned from Alpha Vantage API. Response: {data}")
            return []
            
        # Store in session state for reuse
        st.session_state.news_data[ticker] = data["feed"]
        return data["feed"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news from Alpha Vantage: {e}")
        return []

# Function to fetch market data from Alpha Vantage
def fetch_alpha_vantage_data(ticker, function="TIME_SERIES_DAILY", interval="60min", outputsize="compact"):
    """Fetch market data from Alpha Vantage API"""
    # Determine symbol based on ticker type
    alpha_symbol = TICKERS[ticker]["alpha_symbol"]
    if ":" in alpha_symbol:
        alpha_symbol = alpha_symbol.split(":")[-1]
    
    # Select appropriate function based on ticker type
    if TICKERS[ticker]["type"] == "crypto":
        function = "DIGITAL_CURRENCY_DAILY"
        params = {
            "function": function,
            "symbol": alpha_symbol,
            "market": "USD",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
    elif TICKERS[ticker]["type"] == "forex":
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
    
    # Make API call
    url = "https://www.alphavantage.co/query"
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            st.error(f"Alpha Vantage API error: {data['Error Message']}")
            return pd.DataFrame()
            
        # Parse the response based on function type
        if function == "DIGITAL_CURRENCY_DAILY":
            time_series_key = "Time Series (Digital Currency Daily)"
            if time_series_key not in data:
                st.error(f"Expected data format not found in Alpha Vantage response")
                return pd.DataFrame()
                
            # Create DataFrame from time series data
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                f"1a. open (USD)": "Open",
                f"2a. high (USD)": "High",
                f"3a. low (USD)": "Low",
                f"4a. close (USD)": "Close",
                f"5. volume": "Volume"
            })
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        elif function == "FX_DAILY":
            time_series_key = "Time Series FX (Daily)"
            if time_series_key not in data:
                st.error(f"Expected data format not found in Alpha Vantage response")
                return pd.DataFrame()
                
            # Create DataFrame from time series data
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close"
            })
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        else:  # TIME_SERIES_DAILY
            # Determine which time series key to use
            possible_keys = [
                "Time Series (Daily)",
                "Weekly Time Series",
                "Monthly Time Series"
            ]
            time_series_key = None
            for key in possible_keys:
                if key in data:
                    time_series_key = key
                    break
                    
            if not time_series_key:
                st.error(f"Expected time series data not found in Alpha Vantage response")
                return pd.DataFrame()
                
            # Create DataFrame from time series data
            df = pd.DataFrame(data[time_series_key]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df.sort_index()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error processing Alpha Vantage data: {e}")
        return pd.DataFrame()

# Function to create RAG system
def create_rag_system(ticker, news_articles):
    """Create a RAG system from the fetched news articles"""
    if not news_articles:
        return None
    
    # Create a temporary directory to store article content
    temp_dir = os.path.join(DATA_DIR, ticker)
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
        overall_sentiment_score = "N/A"
        overall_sentiment_label = "N/A"
        ticker_sentiment = "N/A"
        
        if "overall_sentiment_score" in article:
            overall_sentiment_score = article["overall_sentiment_score"]
            overall_sentiment_label = article["overall_sentiment_label"]
            
        if "ticker_sentiment" in article:
            # Try to find sentiment for this specific ticker
            for sentiment in article["ticker_sentiment"]:
                if sentiment["ticker"] == TICKERS[ticker]["alpha_symbol"].split(":")[-1]:
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
TICKER SPECIFIC SENTIMENT ({TICKERS[ticker]['name']}): {ticker_sentiment}

SUMMARY:
{summary}
"""
        
        # Save to a temporary file
        file_path = os.path.join(temp_dir, f"article_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        article_files.append(file_path)
    
    # Create a text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
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
def openrouter_llm_call(prompt, model="mistralai/mistral-7b-instruct", max_tokens=1000):
    """Call the OpenRouter API for LLM tasks"""
    try:
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://financial-insights-hub.streamlit.app",  # Replace with your site
                "X-Title": "Financial Insights Hub",  # Replace with your site name
            },
            model=model,
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
def summarize_daily_news(ticker, news_articles, date_str):
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
                ticker_symbol = TICKERS[ticker]["alpha_symbol"].split(":")[-1]
                for sentiment in article["ticker_sentiment"]:
                    if sentiment.get("ticker") == ticker_symbol:
                        sentiment_info += f", {TICKERS[ticker]['name']} sentiment: {sentiment.get('ticker_sentiment_label', 'N/A')}"
                        break
        
        content += f"TITLE: {title}\nSOURCE: {source}\n{sentiment_info}\n\nSUMMARY: {summary}\n\n---\n\n"
    
    # Create prompt template for summarization
    template = """
    You are a financial analyst specializing in market analysis. Below are news articles about {ticker} from {date}.
    
    {content}
    
    Please provide:
    1. A concise summary of the key developments regarding {ticker} on this date
    2. The overall sentiment (positive, negative, or neutral)
    3. Any potential market impact mentioned
    4. Key factors driving the market sentiment
    
    Keep your analysis focused on factual information from the articles.
    """
    
    # Format the prompt
    formatted_prompt = template.format(
        ticker=TICKERS[ticker]["name"],
        date=date_str,
        content=content
    )
    
    # Call OpenRouter API for the summary
    summary = openrouter_llm_call(
        formatted_prompt, 
        model="mistralai/mistral-7b-instruct",
        max_tokens=1000
    )
    
    return summary

# Function to fetch financial data
def get_financial_data(ticker_symbol, period="1mo"):
    """Fetch financial data for the specified ticker"""
    try:
        data = yf.download(ticker_symbol, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return pd.DataFrame()

# Function to ask questions to the RAG system
def ask_rag(question, vectorstore, ticker):
    """Ask a question to the RAG system"""
    if not vectorstore:
        return "No data available for this ticker. Please try another ticker or refresh the data."
    
    # Get relevant documents from the RAG system
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    
    # Extract context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt template for Q&A
    template = """
    You are an AI assistant specializing in financial analysis for {ticker}.
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
        ticker=TICKERS[ticker]["name"],
        context=context,
        question=question
    )
    
    # Call OpenRouter API for the answer
    answer = openrouter_llm_call(
        formatted_prompt, 
        model="meta-llama/llama-3-70b-instruct",
        max_tokens=1000
    )
    
    return answer

# Function to calculate sentiment score
def calculate_sentiment_score(news_articles):
    """Calculate sentiment score from Alpha Vantage news articles"""
    if not news_articles:
        return 0
        
    total_score = 0
    count = 0
    
    # First try ticker-specific sentiment
    ticker_symbol = None
    for article in news_articles:
        if "ticker_sentiment" not in article:
            continue
            
        for sentiment in article["ticker_sentiment"]:
            ticker = next((k for k, v in TICKERS.items() if v["alpha_symbol"].split(":")[-1] == sentiment.get("ticker")), None)
            if ticker:
                ticker_symbol = sentiment.get("ticker")
                
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
    if count > 0:
        return total_score / count
    else:
        return 0

# Layout functions
def display_ticker_overview(ticker, news_articles):
    """Display overview information for the selected ticker"""
    st.header(f"{TICKERS[ticker]['name']} Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        st.subheader("Price Chart")
        period_options = {"1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year"}
        selected_period = st.selectbox("Select Period", list(period_options.keys()), format_func=lambda x: period_options[x])
        
        data = get_financial_data(TICKERS[ticker]["symbol"], period=selected_period)
        
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
                title=f"{TICKERS[ticker]['name']} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No financial data available from Yahoo Finance. Trying Alpha Vantage...")
            
            # Try getting data from Alpha Vantage
            alpha_data = fetch_alpha_vantage_data(ticker)
            
            if not alpha_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=alpha_data.index,
                    open=alpha_data['Open'],
                    high=alpha_data['High'],
                    low=alpha_data['Low'],
                    close=alpha_data['Close'],
                    name='Price'
                ))
                
                fig.update_layout(
                    title=f"{TICKERS[ticker]['name']} Price Movement",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No financial data available.")
    
    with col2:
        # Current price and stats
        try:
            ticker_data = yf.Ticker(TICKERS[ticker]["symbol"])
            info = ticker_data.info
            
            # Get current price
            live_price = info.get("regularMarketPrice", "N/A")
            prev_close = info.get("regularMarketPreviousClose", "N/A")
            
            if live_price != "N/A" and prev_close != "N/A":
                change = live_price - prev_close
                change_pct = (change / prev_close) * 100
                
                st.metric(
                    label="Current Price", 
                    value=f"${live_price:.2f}", 
                    delta=f"{change:.2f} ({change_pct:.2f}%)"
                )
            else:
                st.metric(label="Current Price", value="N/A")
                
            # Display market cap for stocks/crypto
            if TICKERS[ticker]["type"] in ["crypto", "index", "stock"]:
                market_cap = info.get("marketCap", "N/A")
                if market_cap != "N/A":
                    # Format large numbers
                    if market_cap >= 1_000_000_000_000:
                        market_cap_str = f"${market_cap/1_000_000_000_000:.2f}T"
                    elif market_cap >= 1_000_000_000:
                        market_cap_str = f"${market_cap/1_000_000_000:.2f}B"
                    else:
                        market_cap_str = f"${market_cap/1_000_000:.2f}M"
                    
                    st.metric(label="Market Cap", value=market_cap_str)
            
            # 52-week range for stocks
            if TICKERS[ticker]["type"] in ["index", "commodity", "stock"]:
                week_low = info.get("fiftyTwoWeekLow", "N/A")
                week_high = info.get("fiftyTwoWeekHigh", "N/A")
                
                if week_low != "N/A" and week_high != "N/A":
                    st.metric(label="52-Week Range", value=f"${week_low:.2f} - ${week_high:.2f}")
            
            # Volume
            volume = info.get("regularMarketVolume", "N/A")
            if volume != "N/A":
                if volume >= 1_000_000_000:
                    volume_str = f"{volume/1_000_000_000:.2f}B"
                elif volume >= 1_000_000:
                    volume_str = f"{volume/1_000_000:.2f}M"
                else:
                    volume_str = f"{volume:,}"
                
                st.metric(label="Volume", value=volume_str)
                
        except Exception as e:
            st.warning(f"Error loading ticker data from Yahoo Finance: {e}")
            
        # News sentiment overview
        st.subheader("News Sentiment")
        
        if news_articles:
            # Calculate sentiment from Alpha Vantage news
            avg_sentiment = calculate_sentiment_score(news_articles)
            
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
                    ticker_symbol = TICKERS[ticker]["alpha_symbol"].split(":")[-1]
                    for ts in article["ticker_sentiment"]:
                        if ts.get("ticker") == ticker_symbol:
                            sentiment = ts.get("ticker_sentiment_label", "N/A")
                            break
                
                # If no ticker-specific sentiment, use overall
                if sentiment == "N/A" and "overall_sentiment_label" in article:
                    sentiment = article["overall_sentiment_label"]
                
                # Color-code based on sentiment
                if sentiment == "Bullish" or sentiment == "Somewhat-Bullish":
                    st.success(f"📈 {title} - **{sentiment}**")
                elif sentiment == "Bearish" or sentiment == "Somewhat-Bearish":
                    st.error(f"📉 {title} - **{sentiment}**")
                else:
                    st.info(f"📊 {title} - **{sentiment}**")
        else:
            st.info("No news data available. Please refresh the data.")

def display_news_analysis(ticker, news_articles):
    """Display news analysis for the selected ticker"""
    st.header(f"{TICKERS[ticker]['name']} News Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Daily Summaries", "Latest Articles", "Ask AI Assistant"])
    
    with tab1:
        st.subheader("Daily News Summaries")
        
        if not news_articles:
            st.info("No news articles available.")
            return
            
        # Get available dates from articles
        dates = set()
        for article in news_articles:
            pub_date = article.get("time_published", "")
            if pub_date:
                date_only = pub_date.split("T")[0]  # Extract YYYY-MM-DD
                dates.add(date_only)
        
        dates = sorted(list(dates), reverse=True)
        
        if not dates:
            st.info("No valid publication dates found in the articles.")
            return
            
        selected_date = st.selectbox("Select Date", dates)
        
        # Get or generate summary for selected date
        if ticker not in st.session_state.daily_summaries:
            st.session_state.daily_summaries[ticker] = {}
            
        if selected_date not in st.session_state.daily_summaries[ticker]:
            with st.spinner(f"Generating summary for {selected_date}..."):
                summary = summarize_daily_news(ticker, news_articles, selected_date)
                st.session_state.daily_summaries[ticker][selected_date] = summary
        
        # Display summary
        st.markdown(st.session_state.daily_summaries[ticker][selected_date])
        
        # Button to regenerate summary
        if st.button("Regenerate Summary"):
            with st.spinner("Regenerating summary..."):
                summary = summarize_daily_news(ticker, news_articles, selected_date)
                st.session_state.daily_summaries[summary = summarize_daily_news(ticker, news_articles, selected_date)
                st.session_state.daily_summaries[ticker][selected_date] = summary
    
    with tab2:
        st.subheader("Latest News Articles")
        
        if not news_articles:
            st.info("No news articles available.")
            return
            
        # Display articles with expandable details
        for i, article in enumerate(news_articles[:10]):  # Show top 10 articles
            title = article.get("title", "No title")
            summary = article.get("summary", "No summary available")
            url = article.get("url", "#")
            source = article.get("source", "Unknown source")
            time_published = article.get("time_published", "Unknown date")
            
            # Format the date
            try:
                date_obj = datetime.fromisoformat(time_published.replace("Z", "+00:00"))
                formatted_date = date_obj.strftime("%b %d, %Y %H:%M")
            except:
                formatted_date = time_published
                
            # Get sentiment information
            sentiment_str = ""
            if "overall_sentiment_label" in article:
                sentiment = article["overall_sentiment_label"]
                score = article.get("overall_sentiment_score", "N/A")
                sentiment_str = f"Overall sentiment: {sentiment} ({score})"
                
                # Add ticker-specific sentiment if available
                if "ticker_sentiment" in article:
                    ticker_symbol = TICKERS[ticker]["alpha_symbol"].split(":")[-1]
                    for ts in article["ticker_sentiment"]:
                        if ts.get("ticker") == ticker_symbol:
                            ticker_sentiment = ts.get("ticker_sentiment_label", "N/A")
                            ticker_score = ts.get("ticker_sentiment_score", "N/A")
                            sentiment_str += f" | {TICKERS[ticker]['name']} sentiment: {ticker_sentiment} ({ticker_score})"
                            break
            
            # Color-code based on sentiment
            sentiment_color = "gray"
            if "Bullish" in sentiment_str:
                sentiment_color = "green"
            elif "Bearish" in sentiment_str:
                sentiment_color = "red"
                
            # Create expandable article card
            with st.expander(f"{title} - {formatted_date}"):
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Published:** {formatted_date}")
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment_str}</span>", unsafe_allow_html=True)
                st.markdown("**Summary:**")
                st.markdown(summary)
                st.markdown(f"[Read full article]({url})")
    
    with tab3:
        st.subheader("Ask AI Assistant")
        
        # Create or load RAG system
        if ticker not in st.session_state.rag_index:
            with st.spinner("Building knowledge base from news articles..."):
                st.session_state.rag_index[ticker] = create_rag_system(ticker, news_articles)
        
        # Initialize chat history if needed
        if ticker not in st.session_state.chat_history:
            st.session_state.chat_history[ticker] = []
            
        # Display chat history
        for msg in st.session_state.chat_history[ticker]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
                
        # Get user question
        user_question = st.chat_input("Ask a question about the news...")
        
        if user_question:
            # Add user question to chat history
            st.session_state.chat_history[ticker].append({"role": "user", "content": user_question})
            st.chat_message("user").write(user_question)
            
            # Get answer from RAG system
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.rag_index[ticker]:
                        answer = ask_rag(user_question, st.session_state.rag_index[ticker], ticker)
                    else:
                        answer = "I don't have enough news data to answer questions about this ticker. Please refresh the data."
                    
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history[ticker].append({"role": "assistant", "content": answer})

def display_technical_indicators(ticker):
    """Display technical indicators for the selected ticker"""
    st.header(f"{TICKERS[ticker]['name']} Technical Analysis")
    
    # Fetch data using Alpha Vantage instead of yfinance
    with st.spinner("Loading financial data..."):
        data = fetch_alpha_vantage_data(ticker, function="TIME_SERIES_DAILY")
    
    if data.empty:
        st.error("Could not retrieve financial data for technical analysis.")
        return
        
    # Convert to datetime index if not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Sort data by date
    data = data.sort_index()
    
    # Calculate technical indicators
    # Moving averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_StdDev'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_StdDev']
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_StdDev']
    
    # Create tabs for different indicators
    tab1, tab2, tab3, tab4 = st.tabs(["Moving Averages", "RSI", "MACD", "Bollinger Bands"])
    
    with tab1:
        st.subheader("Moving Averages")
        
        # Create moving averages chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black', width=1)
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA200'],
            mode='lines',
            name='SMA 200',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title=f"{TICKERS[ticker]['name']} - Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Moving average interpretation
        st.subheader("Interpretation")
        
        last_close = data['Close'].iloc[-1]
        last_sma20 = data['SMA20'].iloc[-1]
        last_sma50 = data['SMA50'].iloc[-1]
        last_sma200 = data['SMA200'].iloc[-1]
        
        ma_trend = ""
        if last_close > last_sma20 > last_sma50 > last_sma200:
            ma_trend = "Strong Bullish"
        elif last_close > last_sma20 and last_close > last_sma50:
            ma_trend = "Bullish"
        elif last_close < last_sma20 and last_close < last_sma50:
            ma_trend = "Bearish"
        elif last_close < last_sma20 < last_sma50 < last_sma200:
            ma_trend = "Strong Bearish"
        else:
            ma_trend = "Mixed"
            
        st.markdown(f"**Current Trend:** {ma_trend}")
        st.markdown(f"* Price vs SMA 20: {'Above' if last_close > last_sma20 else 'Below'} (Short-term indicator)")
        st.markdown(f"* Price vs SMA 50: {'Above' if last_close > last_sma50 else 'Below'} (Medium-term indicator)")
        st.markdown(f"* Price vs SMA 200: {'Above' if last_close > last_sma200 else 'Below'} (Long-term indicator)")
        
        if last_sma20 > last_sma50:
            st.markdown("* SMA 20 is above SMA 50, indicating a potential bullish market.")
        else:
            st.markdown("* SMA 20 is below SMA 50, indicating a potential bearish market.")
            
        # Check for golden/death cross
        prev_data = data.iloc[-30:]
        crosses = []
        
        for i in range(1, len(prev_data)):
            # Golden Cross (SMA 50 crosses above SMA 200)
            if (prev_data['SMA50'].iloc[i-1] <= prev_data['SMA200'].iloc[i-1] and 
                prev_data['SMA50'].iloc[i] > prev_data['SMA200'].iloc[i]):
                crosses.append((prev_data.index[i], "Golden Cross"))
                
            # Death Cross (SMA 50 crosses below SMA 200)
            if (prev_data['SMA50'].iloc[i-1] >= prev_data['SMA200'].iloc[i-1] and 
                prev_data['SMA50'].iloc[i] < prev_data['SMA200'].iloc[i]):
                crosses.append((prev_data.index[i], "Death Cross"))
        
        if crosses:
            st.markdown("**Recent Crosses:**")
            for date, cross_type in crosses:
                st.markdown(f"* {cross_type} detected on {date.strftime('%Y-%m-%d')}")
        
    with tab2:
        st.subheader("Relative Strength Index (RSI)")
        
        # Create RSI chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Add horizontal lines for overbought/oversold levels
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=70,
            x1=data.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
        )
        
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=30,
            x1=data.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
        )
        
        fig.update_layout(
            title=f"{TICKERS[ticker]['name']} - Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI interpretation
        st.subheader("Interpretation")
        
        last_rsi = data['RSI'].iloc[-1]
        
        if np.isnan(last_rsi):
            st.warning("Not enough data to calculate RSI.")
        else:
            if last_rsi > 70:
                st.markdown(f"**Current RSI: {last_rsi:.2f} - Overbought**")
                st.markdown("* The asset may be overbought and could be due for a price correction or reversal.")
            elif last_rsi < 30:
                st.markdown(f"**Current RSI: {last_rsi:.2f} - Oversold**")
                st.markdown("* The asset may be oversold and could be due for a price rebound.")
            else:
                st.markdown(f"**Current RSI: {last_rsi:.2f} - Neutral**")
                st.markdown("* The asset is neither overbought nor oversold.")
                
            # Check for RSI divergence (last 20 periods)
            recent_data = data.iloc[-20:].copy()
            if not recent_data.empty and len(recent_data) >= 5:
                price_trend = "up" if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else "down"
                rsi_trend = "up" if recent_data['RSI'].iloc[-1] > recent_data['RSI'].iloc[0] else "down"
                
                if price_trend != rsi_trend:
                    st.markdown("**Potential RSI Divergence Detected:**")
                    if price_trend == "up" and rsi_trend == "down":
                        st.markdown("* Bearish divergence: Price is rising but RSI is falling. This could signal a potential reversal to the downside.")
                    else:
                        st.markdown("* Bullish divergence: Price is falling but RSI is rising. This could signal a potential reversal to the upside.")
    
    with tab3:
        st.subheader("Moving Average Convergence Divergence (MACD)")
        
        # Create MACD chart
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Add price to top subplot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
        
        # Add MACD to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Add MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Hist'],
                name='Histogram',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{TICKERS[ticker]['name']} - MACD",
            xaxis_title="Date",
            height=600
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MACD interpretation
        st.subheader("Interpretation")
        
        last_macd = data['MACD'].iloc[-1]
        last_signal = data['Signal'].iloc[-1]
        last_hist = data['MACD_Hist'].iloc[-1]
        
        if np.isnan(last_macd) or np.isnan(last_signal):
            st.warning("Not enough data to calculate MACD.")
        else:
            # Check current signal
            if last_macd > last_signal:
                st.markdown(f"**Current Signal: Bullish** - MACD ({last_macd:.3f}) is above Signal Line ({last_signal:.3f})")
            else:
                st.markdown(f"**Current Signal: Bearish** - MACD ({last_macd:.3f}) is below Signal Line ({last_signal:.3f})")
                
            # Check histogram
            if last_hist > 0:
                hist_trend = "positive, indicating bullish momentum"
                if last_hist > data['MACD_Hist'].iloc[-2]:
                    hist_trend += " that is strengthening"
                else:
                    hist_trend += " that is weakening"
            else:
                hist_trend = "negative, indicating bearish momentum"
                if last_hist < data['MACD_Hist'].iloc[-2]:
                    hist_trend += " that is strengthening"
                else:
                    hist_trend += " that is weakening"
                    
            st.markdown(f"* MACD Histogram is {hist_trend}.")
            
            # Check for recent crossovers
            recent_data = data.iloc[-30:].copy()
            crossovers = []
            
            for i in range(1, len(recent_data)):
                # Bullish crossover (MACD crosses above Signal)
                if (recent_data['MACD'].iloc[i-1] <= recent_data['Signal'].iloc[i-1] and 
                    recent_data['MACD'].iloc[i] > recent_data['Signal'].iloc[i]):
                    crossovers.append((recent_data.index[i], "Bullish"))
                    
                # Bearish crossover (MACD crosses below Signal)
                if (recent_data['MACD'].iloc[i-1] >= recent_data['Signal'].iloc[i-1] and 
                    recent_data['MACD'].iloc[i] < recent_data['Signal'].iloc[i]):
                    crossovers.append((recent_data.index[i], "Bearish"))
            
            if crossovers:
                st.markdown("**Recent Crossovers:**")
                for date, crossover_type in crossovers:
                    st.markdown(f"* {crossover_type} crossover detected on {date.strftime('%Y-%m-%d')}")
            else:
                st.markdown("* No recent MACD crossovers detected in the last 30 periods.")
    
    with tab4:
        st.subheader("Bollinger Bands")
        
        # Create Bollinger Bands chart
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black', width=1)
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='Upper Band (+2σ)',
            line=dict(color='red', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            mode='lines',
            name='Middle Band (SMA 20)',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='Lower Band (-2σ)',
            line=dict(color='green', width=1)
        ))
        
        # Fill area between upper and lower bands
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            fill='tonexty',
            mode='lines',
            fillcolor='rgba(0, 176, 246, 0.1)',
            line=dict(width=0),
            name='Bands Range'
        ))
        
        fig.update_layout(
            title=f"{TICKERS[ticker]['name']} - Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bollinger Bands interpretation
        st.subheader("Interpretation")
        
        last_close = data['Close'].iloc[-1]
        last_upper = data['BB_Upper'].iloc[-1]
        last_lower = data['BB_Lower'].iloc[-1]
        last_middle = data['BB_Middle'].iloc[-1]
        
        # Calculate bandwidth
        bandwidth = (last_upper - last_lower) / last_middle * 100
        
        # Calculate %B (position within bands)
        percent_b = (last_close - last_lower) / (last_upper - last_lower) * 100
        
        # Interpret current position
        if np.isnan(last_upper) or np.isnan(last_lower):
            st.warning("Not enough data to calculate Bollinger Bands.")
        else:
            if last_close > last_upper:
                st.markdown(f"**Current Position: Above Upper Band** - Price ({last_close:.2f}) is above upper band ({last_upper:.2f})")
                st.markdown("* The asset may be overbought. This could indicate a strong uptrend but also a potential reversal.")
            elif last_close < last_lower:
                st.markdown(f"**Current Position: Below Lower Band** - Price ({last_close:.2f}) is below lower band ({last_lower:.2f})")
                st.markdown("* The asset may be oversold. This could indicate a strong downtrend but also a potential reversal.")
            else:
                st.markdown(f"**Current Position: Within Bands** - Price ({last_close:.2f}) is between upper ({last_upper:.2f}) and lower ({last_lower:.2f}) bands")
                st.markdown("* The asset is trading within expected volatility range.")
                
            st.markdown(f"**%B: {percent_b:.2f}%** - Position within the bands (0% = lower band, 100% = upper band)")
            st.markdown(f"**Bandwidth: {bandwidth:.2f}%** - Indicates market volatility")
            
            # Check for band squeeze (low volatility period)
            recent_data = data.iloc[-30:].copy()
            recent_data['Bandwidth'] = (recent_data['BB_Upper'] - recent_data['BB_Lower']) / recent_data['BB_Middle'] * 100
            avg_bandwidth = recent_data['Bandwidth'].mean()
            
            if bandwidth < avg_bandwidth * 0.8:
                st.markdown("* **Bollinger Band Squeeze detected** - Low volatility period often precedes a significant price movement.")
            elif bandwidth > avg_bandwidth * 1.5:
                st.markdown("* **High volatility period** - Market is experiencing larger than normal price movements.")

def refresh_all_data():
    """Refresh all cached data for the selected ticker"""
    ticker = st.session_state.current_ticker
    
    with st.spinner(f"Refreshing data for {TICKERS[ticker]['name']}..."):
        # Fetch news data
        news_articles = fetch_alpha_vantage_news(ticker)
        
        if news_articles:
            st.session_state.news_data[ticker] = news_articles
            
            # Reset RAG index
            if ticker in st.session_state.rag_index:
                del st.session_state.rag_index[ticker]
                
            # Reset daily summaries
            if ticker in st.session_state.daily_summaries:
                del st.session_state.daily_summaries[ticker]
                
            # Update last updated time
            st.session_state.last_updated[ticker] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            st.success(f"Data refreshed successfully! ({len(news_articles)} articles found)")
        else:
            st.error("Failed to fetch news data. Please try again later.")

# Main application layout
def main():
    # Add custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .sentiment-positive {
        color: green;
    }
    .sentiment-negative {
        color: red;
    }
    .sentiment-neutral {
        color: gray;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Financial Insights Hub")
    st.sidebar.image("https://img.icons8.com/color/96/000000/financial-growth-analysis.png", width=100)
    
    # Ticker selection
    ticker_options = {k: v["name"] for k, v in TICKERS.items()}
    selected_ticker = st.sidebar.selectbox("Select Asset", list(ticker_options.keys()), format_func=lambda x: ticker_options[x])
    
    # Store current ticker in session state
    if "current_ticker" not in st.session_state or st.session_state.current_ticker != selected_ticker:
        st.session_state.current_ticker = selected_ticker
    
    # View selection
    views = ["Overview", "News Analysis", "Technical Analysis"]
    selected_view = st.sidebar.radio("Select View", views)
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data"):
        refresh_all_data()
        
    # Show last updated time
    if selected_ticker in st.session_state.last_updated:
        st.sidebar.info(f"Last updated: {st.session_state.last_updated[selected_ticker]}")
    
    # Get news data for the selected ticker (fetch if not in session)
    if selected_ticker not in st.session_state.news_data:
        with st.spinner(f"Fetching news data for {TICKERS[selected_ticker]['name']}..."):
            news_articles = fetch_alpha_vantage_news(selected_ticker)
            if news_articles:
                st.session_state.news_data[selected_ticker] = news_articles
                st.session_state.last_updated[selected_ticker] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    news_articles = st.session_state.news_data.get(selected_ticker, [])
    
    # Display the selected view
    if selected_view == "Overview":
        display_ticker_overview(selected_ticker, news_articles)
    elif selected_view == "News Analysis":
        display_news_analysis(selected_ticker, news_articles)
    elif selected_view == "Technical Analysis":
        display_technical_indicators(selected_ticker)
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Data provided by Alpha Vantage API")
    st.sidebar.caption("© 2025 Financial Insights Hub")

if __name__ == "__main__":
    main()