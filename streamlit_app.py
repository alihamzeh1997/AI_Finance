# app.py - Main Streamlit Application
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
# from langchain_community.llms import OpenRouter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
import plotly.graph_objs as go
import plotly.express as px

# Configuration and API Keys
GUARDIAN_API_KEY = st.secrets["GUARDIAN_API_KEY"]  # Set this in Streamlit secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]  # Set this in Streamlit secrets

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

# Define ticker mapping (symbol to readable name and Yahoo Finance symbol)
TICKERS = {
    "bitcoin": {"name": "Bitcoin (BTC)", "symbol": "BTC-USD", "type": "crypto"},
    "ethereum": {"name": "Ethereum (ETH)", "symbol": "ETH-USD", "type": "crypto"},
    "gold": {"name": "Gold", "symbol": "GC=F", "type": "commodity"},
    "silver": {"name": "Silver", "symbol": "SI=F", "type": "commodity"},
    "sp500": {"name": "S&P 500", "symbol": "^GSPC", "type": "index"},
    "dow": {"name": "Dow Jones", "symbol": "^DJI", "type": "index"},
    "nasdaq": {"name": "NASDAQ", "symbol": "^IXIC", "type": "index"},
    "eurusd": {"name": "EUR/USD", "symbol": "EURUSD=X", "type": "forex"},
    "usdjpy": {"name": "USD/JPY", "symbol": "JPY=X", "type": "forex"},
}

# Configure data directory for storing article data
DATA_DIR = "financial_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Function to fetch articles from Guardian API
def fetch_guardian_articles(ticker, days_back=7):
    """Fetch articles related to the ticker from The Guardian API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for Guardian API
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    # Prepare search terms based on ticker
    if ticker == "bitcoin":
        search_terms = "bitcoin OR btc OR cryptocurrency"
    elif ticker == "ethereum":
        search_terms = "ethereum OR eth OR cryptocurrency"
    elif ticker == "gold":
        search_terms = "gold price OR gold market"
    elif ticker == "silver":
        search_terms = "silver price OR silver market"
    elif ticker == "sp500":
        search_terms = "S&P 500 OR S&P500 OR Standard and Poor's 500"
    elif ticker == "dow":
        search_terms = "Dow Jones OR DJIA"
    elif ticker == "nasdaq":
        search_terms = "NASDAQ OR Nasdaq Composite"
    elif ticker == "eurusd":
        search_terms = "EUR/USD OR Euro Dollar exchange"
    elif ticker == "usdjpy":
        search_terms = "USD/JPY OR Dollar Yen exchange"
    else:
        search_terms = ticker
    
    base_url = "https://content.guardianapis.com/search"
    
    params = {
        "q": search_terms,
        "from-date": from_date,
        "to-date": to_date,
        "order-by": "newest",
        "show-fields": "body,headline,byline,wordcount",
        "page-size": 50,
        "api-key": GUARDIAN_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        data = response.json()
        return data.get("response", {}).get("results", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching articles from The Guardian: {e}")
        return []

# Function to create RAG system
def create_rag_system(ticker, articles):
    """Create a RAG system from the fetched articles"""
    if not articles:
        return None
    
    # Create a temporary directory to store article content
    temp_dir = os.path.join(DATA_DIR, ticker)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save articles to text files
    article_files = []
    for i, article in enumerate(articles):
        fields = article.get("fields", {})
        title = fields.get("headline", "No title")
        body = fields.get("body", "No content")
        date = article.get("webPublicationDate", "No date")
        
        # Combine title and body for the document
        content = f"TITLE: {title}\nDATE: {date}\n\n{body}"
        
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

# Function to summarize articles
def summarize_daily_news(ticker, articles, date_str):
    """Generate a summary for the ticker for a specific date using OpenRouter"""
    
    # Filter articles for the specified date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = date_obj + timedelta(days=1)
    date_articles = [
        article for article in articles 
        if date_str in article.get("webPublicationDate", "")
    ]
    
    if not date_articles:
        return "No articles found for this date."
    
    # Prepare content for summarization
    content = ""
    for article in date_articles[:5]:  # Limit to 5 articles to avoid token limits
        fields = article.get("fields", {})
        title = fields.get("headline", "No title")
        body = fields.get("body", "")
        # Take first 1000 characters of body to stay within token limits
        content += f"TITLE: {title}\n\nCONTENT: {body[:1000]}\n\n---\n\n"
    
    # Create the LLM for summarization (using a lighter model)
    llm = OpenRouter(
        api_key=OPENROUTER_API_KEY,
        model="mistralai/mistral-7b-instruct",  # Using a smaller model for summarization
        max_tokens=1000
    )
    
    # Create prompt template for summarization
    template = """
    You are a financial analyst specializing in market analysis. Below are news articles about {ticker} from {date}.
    
    {content}
    
    Please provide:
    1. A concise summary of the key developments regarding {ticker} on this date
    2. The overall sentiment (positive, negative, or neutral)
    3. Any potential market impact mentioned
    
    Keep your analysis focused on factual information from the articles.
    """
    
    prompt = PromptTemplate(
        input_variables=["ticker", "date", "content"],
        template=template
    )
    
    # Format the prompt
    formatted_prompt = prompt.format(
        ticker=TICKERS[ticker]["name"],
        date=date_str,
        content=content
    )
    
    try:
        # Generate summary
        summary = llm.invoke(formatted_prompt)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Error generating summary. Please try again later."

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
    
    # Create the LLM (using a more powerful model for Q&A)
    llm = OpenRouter(
        api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3-70b-instruct",  # More powerful model for RAG Q&A
        max_tokens=1000
    )
    
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
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "ticker"],
        template=template
    )
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    
    # Invoke the chain
    try:
        result = qa({"query": question, "ticker": TICKERS[ticker]["name"]})
        return result["result"]
    except Exception as e:
        st.error(f"Error querying the RAG system: {e}")
        return "Error processing your question. Please try again later."

# Function to calculate sentiment score
def calculate_sentiment_score(summary):
    """Extract sentiment from summary"""
    # This is a simple rule-based approach - in production you'd use NLP
    positive_words = ["positive", "bullish", "gain", "rise", "increase", "growth", "up"]
    negative_words = ["negative", "bearish", "loss", "fall", "decrease", "decline", "down"]
    
    summary_lower = summary.lower()
    
    positive_count = sum(1 for word in positive_words if word in summary_lower)
    negative_count = sum(1 for word in negative_words if word in summary_lower)
    
    # Calculate score between -1 and 1
    if positive_count + negative_count == 0:
        return 0
    return (positive_count - negative_count) / (positive_count + negative_count)

# Layout functions
def display_ticker_overview(ticker):
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
            if TICKERS[ticker]["type"] in ["crypto", "index"]:
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
            if TICKERS[ticker]["type"] in ["index", "commodity"]:
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
            st.error(f"Error loading ticker data: {e}")
            
        # News sentiment overview
        st.subheader("News Sentiment")
        
        if ticker in st.session_state.daily_summaries:
            # Calculate average sentiment across days
            sentiments = []
            for date, summary in st.session_state.daily_summaries[ticker].items():
                sentiment = calculate_sentiment_score(summary)
                sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                
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
            else:
                st.info("No sentiment data available.")
        else:
            st.info("No news summaries available. Please refresh the data.")

def display_news_analysis(ticker, articles):
    """Display news analysis for the selected ticker"""
    st.header(f"{TICKERS[ticker]['name']} News Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Daily Summaries", "Latest Articles", "Ask AI Assistant"])
    
    with tab1:
        st.subheader("Daily News Summaries")
        
        # Get available dates from articles
        dates = set()
        for article in articles:
            pub_date = article.get("webPublicationDate", "")
            if pub_date:
                date_only = pub_date.split("T")[0]  # Extract YYYY-MM-DD
                dates.add(date_only)
        
        dates = sorted(list(dates), reverse=True)
        
        if not dates:
            st.info("No news articles available.")
            return
            
        selected_date = st.selectbox("Select Date", dates)
        
        # Get or generate summary for selected date
        if ticker not in st.session_state.daily_summaries:
            st.session_state.daily_summaries[ticker] = {}
            
        if selected_date not in st.session_state.daily_summaries[ticker]:
            with st.spinner(f"Generating summary for {selected_date}..."):
                summary = summarize_daily_news(ticker, articles, selected_date)
                st.session_state.daily_summaries[ticker][selected_date] = summary
        
        # Display summary
        st.markdown(st.session_state.daily_summaries[ticker][selected_date])
        
        # Button to regenerate summary
        if st.button("Regenerate Summary"):
            with st.spinner("Regenerating summary..."):
                summary = summarize_daily_news(ticker, articles, selected_date)
                st.session_state.daily_summaries[ticker][selected_date] = summary
                st.experimental_rerun()
    
    with tab2:
        st.subheader("Latest News Articles")
        
        for article in articles[:10]:  # Show most recent 10 articles
            title = article.get("fields", {}).get("headline", "No title")
            date = article.get("webPublicationDate", "No date").replace("T", " ").split("+")[0]
            url = article.get("webUrl", "#")
            
            st.markdown(f"#### [{title}]({url})")
            st.caption(f"Published: {date}")
            
            with st.expander("View Article Preview"):
                body = article.get("fields", {}).get("body", "No content available")
                # Show first 500 characters as preview
                st.write(body[:500] + "..." if len(body) > 500 else body)
                st.markdown(f"[Read full article]({url})")
            
            st.divider()
    
    with tab3:
        st.subheader("Ask AI Assistant")
        
        # Initialize chat history for this ticker if it doesn't exist
        if ticker not in st.session_state.chat_history:
            st.session_state.chat_history[ticker] = []
        
        # Display chat history
        for message in st.session_state.chat_history[ticker]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about " + TICKERS[ticker]["name"]):
            st.chat_message("user").write(prompt)
            
            # Add to history
            st.session_state.chat_history[ticker].append({"role": "user", "content": prompt})
            
            # Get response from RAG system
            vectorstore = st.session_state.rag_index.get(ticker)
            
            if not vectorstore:
                response = "I don't have enough information about this ticker yet. Please refresh the data."
            else:
                with st.spinner("Thinking..."):
                    response = ask_rag(prompt, vectorstore, ticker)
            
            # Display response
            st.chat_message("assistant").write(response)
            
            # Add to history
            st.session_state.chat_history[ticker].append({"role": "assistant", "content": response})

# Main application
def main():
    st.title("Financial Insights Hub")
    
    # Sidebar
    st.sidebar.header("Navigation")
    
    # Select ticker
    ticker = st.sidebar.selectbox(
        "Select Financial Asset",
        list(TICKERS.keys()),
        format_func=lambda x: TICKERS[x]["name"]
    )
    
    # Refresh data button
    refresh_col1, refresh_col2 = st.sidebar.columns([3, 1])
    with refresh_col1:
        st.write("Last Updated:")
        if ticker in st.session_state.last_updated:
            st.write(st.session_state.last_updated[ticker].strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.write("Never")
    
    with refresh_col2:
        refresh = st.button("Refresh Data")
    
    # Fetch and process data
    if refresh or ticker not in st.session_state.rag_index:
        with st.spinner(f"Fetching latest data for {TICKERS[ticker]['name']}..."):
            # Fetch articles
            articles = fetch_guardian_articles(ticker)
            
            if articles:
                # Create RAG system
                vectorstore = create_rag_system(ticker, articles)
                if vectorstore:
                    st.session_state.rag_index[ticker] = vectorstore
                    st.session_state.last_updated[ticker] = datetime.now()
                    
                    # Reset daily summaries for this ticker
                    st.session_state.daily_summaries[ticker] = {}
                    
                    st.success(f"Successfully updated data for {TICKERS[ticker]['name']}")
                else:
                    st.error("Failed to create RAG system. Try again.")
            else:
                st.error("No articles found. Check API key or try different search terms.")
    
    # Main content
    articles = []
    if ticker in st.session_state.rag_index:
        # Fetch articles again just for display (we already have the RAG system)
        articles = fetch_guardian_articles(ticker)
    
    tab1, tab2 = st.tabs(["Overview", "News Analysis"])
    
    with tab1:
        display_ticker_overview(ticker)
    
    with tab2:
        display_news_analysis(ticker, articles)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Financial Insights Hub - Powered by The Guardian API and OpenRouter")

if __name__ == "__main__":
    main()