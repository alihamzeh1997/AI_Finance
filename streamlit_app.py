# app.py - Financial Insights Hub (Stocks Only)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import time
from datetime import datetime, timedelta
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
    page_title="Financial Insights Hub - Stocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
def init_session_state():
    session_vars = {
        "daily_summaries": {},
        "rag_index": {},
        "last_updated": {},
        "chat_history": {},
        "news_data": {},
        "predictions": {}
    }
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = session_vars[var]

init_session_state()

# Define stock tickers
TICKERS = {
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
}

# Alpha Vantage API Utilities
class AlphaVantageAPI:
    BASE_URL = "https://www.alphavantage.co/query"
    
    @staticmethod
    def fetch_news(asset_symbol, start_date=None, end_date=None, limit=50):
        alpha_symbol = TICKERS[asset_symbol]["alpha_symbol"]
        
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "earnings,ipo,mergers_and_acquisitions,financial_markets",
            "tickers": alpha_symbol,
            "time_from": (datetime.now() - timedelta(days=7)).strftime("%Y%m%dT0000") if start_date is None else start_date.strftime("%Y%m%dT0000"),
            "limit": limit,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        if end_date:
            params["time_to"] = end_date.strftime("%Y%m%dT2359")
        
        try:
            response = requests.get(AlphaVantageAPI.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data.get("feed", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching news: {e}")
            return []

    @staticmethod
    def fetch_market_data(asset_symbol, outputsize="compact"):
        alpha_symbol = TICKERS[asset_symbol]["alpha_symbol"]
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": alpha_symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": outputsize
        }
        
        try:
            response = requests.get(AlphaVantageAPI.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                st.error(f"API Error: {data['Error Message']}")
                return pd.DataFrame()
            
            return AlphaVantageAPI._parse_market_data(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _parse_market_data(data):
        time_series_key = "Time Series (Daily)"
        rename_cols = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }
        
        time_series = data.get(time_series_key, {})
        if not time_series:
            st.error("No time series data found in response")
            return pd.DataFrame()
        
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        
        # Rename columns to standard names
        df = df.rename(columns=rename_cols)
        
        # Ensure all expected columns exist
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        return df.apply(pd.to_numeric, errors="coerce").sort_index()

# LLM Utilities
class LLMAPI:
    @staticmethod
    def call(prompt, max_tokens=1000):
        try:
            completion = openrouter_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://financial-insights-hub.streamlit.app",
                    "X-Title": "Financial Insights Hub",
                },
                model="nvidia/llama-3.1-nemotron-70b-instruct:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"LLM Error: {e}")
            return "Error generating response"

# Financial Analysis Utilities
class FinancialAnalyzer:
    @staticmethod
    def calculate_sentiment(news_articles, asset_symbol):
        ticker_symbol = TICKERS[asset_symbol]["alpha_symbol"]
        total_score = 0
        count = 0
        
        for article in news_articles:
            if "ticker_sentiment" in article:
                for sentiment in article["ticker_sentiment"]:
                    if sentiment.get("ticker") == ticker_symbol and "ticker_sentiment_score" in sentiment:
                        total_score += float(sentiment["ticker_sentiment_score"])
                        count += 1
            
            elif "overall_sentiment_score" in article:
                total_score += float(article["overall_sentiment_score"])
                count += 1
        
        return total_score / count if count > 0 else 0
    
    @staticmethod
    def create_rag_system(asset_symbol, news_articles):
        if not news_articles:
            return None
        
        temp_dir = os.path.join("financial_data", f"stock_{asset_symbol}")
        os.makedirs(temp_dir, exist_ok=True)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = []
        
        for i, article in enumerate(news_articles):
            content = f"""TITLE: {article.get("title", "No title")}
DATE: {article.get("time_published", "No date")}
SOURCE: {article.get("source", "Unknown source")}
AUTHORS: {', '.join(article.get("authors", [])) if article.get("authors") else 'Unknown'}
URL: {article.get("url", "No URL")}
OVERALL SENTIMENT: {article.get("overall_sentiment_label", "N/A")}
SUMMARY: {article.get("summary", "No content")}
"""
            file_path = os.path.join(temp_dir, f"article_{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            loader = TextLoader(file_path, encoding="utf-8")
            docs = text_splitter.split_documents(loader.load())
            documents.extend(docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        return FAISS.from_documents(documents, embeddings)
    
    @staticmethod
    def ask_rag(question, vectorstore, asset_name):
        if not vectorstore:
            return "No data available for this asset."
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        template = f"""
        You are an AI assistant specializing in financial analysis for {asset_name}.
        Use the following context to answer the question:
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        
        return LLMAPI.call(template, max_tokens=1000)
    
    @staticmethod
    def predict_price(asset_name, data, sentiment_score):
        if data.empty or len(data) < 50:
            return "Insufficient data for prediction"
        
        df = data.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        latest = df.iloc[-1]
        short_term_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
        
        template = f"""
        Predict tomorrow's price for {asset_name} based on:
        - Current price: ${latest['Close']:.2f}
        - 5-day change: {short_term_change:.2f}%
        - RSI (14): {latest.get('RSI_14', np.nan):.2f}
        - MACD: {latest.get('MACD_12_26_9', np.nan):.4f}
        - Bollinger Band Upper: ${latest.get('BBU_20_2.0', np.nan):.2f}
        - Bollinger Band Lower: ${latest.get('BBL_20_2.0', np.nan):.2f}
        - Sentiment Score: {sentiment_score:.2f}
        
        Provide:
        1. Specific price prediction
        2. Reasoning
        3. Confidence level
        4. Key influencing factors
        """
        
        return LLMAPI.call(template, max_tokens=800)

# Streamlit UI Components
class UIComponents:
    @staticmethod
    def display_home_page():
        st.title("🌟 Financial Insights Hub - Stocks")
        st.markdown("""
        ### Welcome to your comprehensive stock analysis platform!
        
        This application provides real-time financial insights, news analysis, and technical indicators for major stocks.
        Powered by Alpha Vantage data and advanced AI analysis, it helps you make informed financial decisions.
        
        #### 🔑 Key Features:
        - **Stock Analysis**: Analyze major stocks
        - **News Sentiment Analysis**: Track market sentiment from financial news
        - **Technical Indicators**: View key technical analysis metrics
        - **AI-Powered Insights**: Ask questions about market trends
        - **Price Predictions**: AI-generated price forecasts
        - **Custom Date Ranges**: Select specific timeframes for your analysis
        
        #### 📊 Getting Started:
        1. Choose a stock from the sidebar
        2. Select your desired date range for analysis
        3. Explore different views: Overview, News Analysis, or Technical Analysis
        4. Use the AI assistant to ask specific questions about your selected stock
        """)
        
        # Sample visualization
        st.subheader("Market Sentiment Overview")
        stocks = ["Apple", "Microsoft", "Amazon", "Tesla", "NVIDIA"]
        sentiments = [0.45, 0.12, 0.35, -0.25, 0.32]
        fig = px.bar(
            x=stocks,
            y=sentiments,
            color=sentiments,
            color_continuous_scale=["red", "gray", "green"],
            range_color=[-1, 1],
            labels={"x": "Stock", "y": "Market Sentiment"},
            title="Current Market Sentiment for Major Stocks"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_overview(asset_symbol, news_articles, data):
        asset_info = TICKERS[asset_symbol]
        st.header(f"{asset_info['name']} Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not data.empty:
                fig = go.Figure(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                ))
                fig.update_layout(
                    title=f"{asset_info['name']} Price Movement",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No financial data available")
        
        with col2:
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                change = latest["Close"] - prev["Close"]
                change_pct = (change / prev["Close"]) * 100
                
                st.metric(
                    label="Current Price", 
                    value=f"${latest['Close']:.2f}", 
                    delta=f"{change:.2f} ({change_pct:.2f}%)"
                )
                
                volume = latest["Volume"]
                volume_str = f"{volume/1e6:.2f}M" if volume >= 1e6 else f"{volume/1e3:.2f}K"
                st.metric(label="Volume", value=volume_str)
                
                try:
                    year_data = data.last('365D')
                    st.metric(
                        label="52-Week Range", 
                        value=f"${year_data['Low'].min():.2f} - ${year_data['High'].max():.2f}"
                    )
                except:
                    pass
            
            if news_articles:
                avg_sentiment = FinancialAnalyzer.calculate_sentiment(news_articles, asset_symbol)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_sentiment,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "News Sentiment"},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'steps': [
                            {'range': [-1, -0.33], 'color': "red"},
                            {'range': [-0.33, 0.33], 'color': "gray"},
                            {'range': [0.33, 1], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                for article in news_articles[:3]:
                    title = article.get("title", "No title")
                    sentiment = next(
                        (ts.get("ticker_sentiment_label", "N/A") for ts in article.get("ticker_sentiment", []) 
                         if ts.get("ticker") == asset_info["alpha_symbol"]),
                        article.get("overall_sentiment_label", "N/A")
                    )
                    
                    color_map = {"bullish": "green", "positive": "green", "bearish": "red", "negative": "red"}
                    color = color_map.get(sentiment.lower(), "gray")
                    st.markdown(f"<span style='color:{color}; font-weight:bold'>•</span> **{title}** - *{sentiment}*", unsafe_allow_html=True)
        
        if data.empty or not news_articles:
            st.warning("Insufficient data for price prediction")
        else:
            st.subheader("AI Price Prediction")
            key = f"stock_{asset_symbol}"
            if key not in st.session_state.predictions:
                sentiment_score = FinancialAnalyzer.calculate_sentiment(news_articles, asset_symbol)
                prediction = FinancialAnalyzer.predict_price(asset_info["name"], data, sentiment_score)
                st.session_state.predictions[key] = prediction
            st.markdown(st.session_state.predictions[key])

    @staticmethod
    def display_news_analysis(asset_symbol, news_articles, start_date, end_date):
        asset_info = TICKERS[asset_symbol]
        st.header(f"{asset_info['name']} News Analysis")
        
        if not news_articles:
            st.warning("No news articles available")
            return
        
        # Organize news by date
        news_by_date = {}
        for article in news_articles:
            date_str = article.get("time_published", "")[:10]
            if date_str:
                news_by_date.setdefault(date_str, []).append(article)
        
        # Display news volume
        st.subheader("News Volume by Date")
        date_counts = {date: len(articles) for date, articles in news_by_date.items()}
        fig = px.bar(x=list(date_counts.keys()), y=list(date_counts.values()))
        st.plotly_chart(fig, use_container_width=True)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Daily Summaries", "All Articles", "Ask about News"])
        
        with tab1:
            for date in sorted(news_by_date.keys(), reverse=True):
                with st.expander(f"Summary for {date}"):
                    key = f"{asset_symbol}_{date}"
                    if key not in st.session_state.daily_summaries:
                        with st.spinner("Generating summary..."):
                            summary = LLMAPI.call(
                                f"Summarize the key developments for {asset_info['name']} on {date} from these articles: "
                                + "\n\n".join([a.get("summary", "") for a in news_by_date[date][:5]]),
                                max_tokens=500
                            )
                            st.session_state.daily_summaries[key] = summary
                    st.markdown(st.session_state.daily_summaries[key])
        
        with tab2:
            st.subheader("All News Articles")
            articles_data = []
            for article in news_articles:
                sentiment = next(
                    (ts.get("ticker_sentiment_label", "N/A") for ts in article.get("ticker_sentiment", []) 
                     if ts.get("ticker") == asset_info["alpha_symbol"]),
                    article.get("overall_sentiment_label", "N/A")
                )
                articles_data.append({
                    "Date": article.get("time_published", "")[:10],
                    "Source": article.get("source", "Unknown"),
                    "Title": article.get("title", "No title"),
                    "Sentiment": sentiment,
                    "URL": article.get("url", "#")
                })
            
            df = pd.DataFrame(articles_data)
            st.dataframe(
                df.style.applymap(
                    lambda x: 'background-color: rgba(0, 128, 0, 0.2)' if x.lower() in ['bullish', 'positive'] 
                    else 'background-color: rgba(255, 0, 0, 0.2)' if x.lower() in ['bearish', 'negative'] 
                    else '', subset=['Sentiment']
                ),
                column_config={"URL": st.column_config.LinkColumn("Link")},
                hide_index=True,
                use_container_width=True
            )
        
        with tab3:
            key = f"stock_{asset_symbol}"
            if key not in st.session_state.rag_index:
                with st.spinner("Building knowledge base..."):
                    st.session_state.rag_index[key] = FinancialAnalyzer.create_rag_system(asset_symbol, news_articles)
            
            user_question = st.text_input("Ask a question about the news:", key=f"question_{key}")
            if user_question:
                if key not in st.session_state.chat_history:
                    st.session_state.chat_history[key] = []
                st.session_state.chat_history[key].append({"role": "user", "content": user_question})
                
                with st.spinner("Analyzing..."):
                    answer = FinancialAnalyzer.ask_rag(user_question, st.session_state.rag_index[key], asset_info["name"])
                    st.session_state.chat_history[key].append({"role": "assistant", "content": answer})
            
            if key in st.session_state.chat_history:
                for msg in st.session_state.chat_history[key]:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")
                    st.markdown("---")

    @staticmethod
    def display_technical_analysis(asset_symbol, data):
        asset_info = TICKERS[asset_symbol]
        st.header(f"{asset_info['name']} Technical Analysis")
        
        if data.empty:
            st.error("No data available for analysis")
            return
        
        df = data.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        
        tab1, tab2, tab3 = st.tabs(["Price & Moving Averages", "Oscillators", "Advanced Analysis"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'))
            st.plotly_chart(fig, use_container_width=True)
            
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("SMA 20", f"${latest['SMA_20']:.2f}", f"{((current_price/latest['SMA_20'])-1)*100:.2f}%")
            col2.metric("SMA 50", f"${latest['SMA_50']:.2f}", f"{((current_price/latest['SMA_50'])-1)*100:.2f}%")
            col3.metric("SMA 200", f"${latest['SMA_200']:.2f}", f"{((current_price/latest['SMA_200'])-1)*100:.2f}%")
            
            if latest['SMA_50'] > latest['SMA_200']:
                st.success("Bullish Trend: 50-day SMA above 200-day SMA")
            else:
                st.error("Bearish Trend: 50-day SMA below 200-day SMA")
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI'))
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
            fig.update_layout(title="RSI (14)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            latest_rsi = df['RSI_14'].iloc[-1]
            if latest_rsi > 70:
                st.warning(f"Overbought: RSI at {latest_rsi:.2f}")
            elif latest_rsi < 30:
                st.success(f"Oversold: RSI at {latest_rsi:.2f}")
            else:
                st.info(f"Neutral RSI: {latest_rsi:.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal'))
            fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram'))
            fig.update_layout(title="MACD (12,26,9)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            latest_macd = df['MACD_12_26_9'].iloc[-1]
            latest_signal = df['MACDs_12_26_9'].iloc[-1]
            if latest_macd > latest_signal:
                st.success("Bullish MACD Signal")
            else:
                st.error("Bearish MACD Signal")
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='Upper BB'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBM_20_2.0'], name='Middle BB'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='Lower BB'))
            fig.update_layout(title="Bollinger Bands (20,2)", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            latest = df.iloc[-1]
            bandwidth = (latest['BBU_20_2.0'] - latest['BBL_20_2.0']) / latest['BBM_20_2.0'] * 100
            percent_b = (latest['Close'] - latest['BBL_20_2.0']) / (latest['BBU_20_2.0'] - latest['BBL_20_2.0']) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Bandwidth", f"{bandwidth:.2f}%")
            col2.metric("%B", f"{percent_b:.2f}%")
            
            if latest['Close'] > latest['BBU_20_2.0']:
                st.warning("Price above Upper Bollinger Band")
            elif latest['Close'] < latest['BBL_20_2.0']:
                st.success("Price below Lower Bollinger Band")
            else:
                st.info("Price within Bollinger Bands")

# Main Application
def main():
    st.sidebar.title("Financial Insights Hub - Stocks")
    selected_page = st.sidebar.radio("Navigation", ["Home", "Stock Analysis"])
    
    if selected_page == "Home":
        UIComponents.display_home_page()
    else:
        # Stock selection
        st.sidebar.header("Stock Selection")
        asset_symbol = st.sidebar.selectbox(
            "Select Stock",
            list(TICKERS.keys()),
            format_func=lambda x: f"{x} - {TICKERS[x]['name']}"
        )
        
        # Date range
        st.sidebar.header("Date Range")
        today = datetime.now().date()
        start_date = st.sidebar.date_input("Start Date", today - timedelta(days=30))
        end_date = st.sidebar.date_input("End Date", today)
        
        # Refresh data
        if st.sidebar.button("Refresh Data"):
            st.session_state.pop("last_updated", None)
        
        # Fetch data
        data_key = f"stock_{asset_symbol}_data"
        news_key = f"stock_{asset_symbol}_news"
        
        if data_key not in st.session_state or news_key not in st.session_state:
            with st.spinner("Loading data..."):
                market_data = AlphaVantageAPI.fetch_market_data(asset_symbol)
                news_data = AlphaVantageAPI.fetch_news(asset_symbol, start_date, end_date)
                st.session_state[data_key] = market_data
                st.session_state[news_key] = news_data
                st.session_state["last_updated"] = datetime.now()
        
        # Display tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "News Analysis", "Technical Analysis"])
        
        with tab1:
            UIComponents.display_overview(
                asset_symbol, 
                st.session_state[news_key], 
                st.session_state[data_key]
            )
        
        with tab2:
            UIComponents.display_news_analysis(
                asset_symbol, 
                st.session_state[news_key], 
                start_date, 
                end_date
            )
        
        with tab3:
            UIComponents.display_technical_analysis(
                asset_symbol, 
                st.session_state[data_key]
            )

if __name__ == "__main__":
    main()