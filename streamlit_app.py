# app.py - Optimized Main Streamlit Application with Alpha Vantage
import streamlit as st
import pandas as pd
import numpy as np
import requests, os, json, time
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import pandas_ta as ta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from plotly.subplots import make_subplots

# Configuration
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
st.set_page_config(page_title="Financial Insights Hub", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
for key in ["daily_summaries", "rag_index", "last_updated", "chat_history", "news_data", "predictions"]:
    st.session_state.setdefault(key, {})

# Ticker data
TICKERS = {
    "crypto": {
        "BTC": {"name": "Bitcoin", "alpha_symbol": "CRYPTO:BTC"},
        "ETH": {"name": "Ethereum", "alpha_symbol": "CRYPTO:ETH"},
        "BNB": {"name": "Binance Coin", "alpha_symbol": "CRYPTO:BNB"},
        "SOL": {"name": "Solana", "alpha_symbol": "CRYPTO:SOL"},
        "ADA": {"name": "Cardano", "alpha_symbol": "CRYPTO:ADA"}
    },
    "stock": {
        "AAPL": {"name": "Apple Inc.", "alpha_symbol": "AAPL"},
        "MSFT": {"name": "Microsoft", "alpha_symbol": "MSFT"},
        "GOOGL": {"name": "Alphabet", "alpha_symbol": "GOOGL"},
        "AMZN": {"name": "Amazon", "alpha_symbol": "AMZN"},
        "META": {"name": "Meta Platforms", "alpha_symbol": "META"}
    },
    "index": {
        "SPX": {"name": "S&P 500", "alpha_symbol": "INDEX:SPX"},
        "DJI": {"name": "Dow Jones", "alpha_symbol": "INDEX:DJI"},
        "COMP": {"name": "NASDAQ", "alpha_symbol": "INDEX:COMP"}
    },
    "forex": {
        "EUR/USD": {"name": "Euro/US Dollar", "alpha_symbol": "FOREX:EUR/USD"},
        "USD/JPY": {"name": "US Dollar/Japanese Yen", "alpha_symbol": "FOREX:USD/JPY"}
    },
    "commodity": {
        "GOLD": {"name": "Gold", "alpha_symbol": "COMMODITY:GOLD"},
        "OIL": {"name": "Crude Oil", "alpha_symbol": "COMMODITY:OIL"}
    }
}

# Data directory setup
DATA_DIR = "financial_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Helper functions
def get_alpha_symbol(asset_category, asset_symbol):
    return TICKERS[asset_category][asset_symbol]["alpha_symbol"].split(":")[-1]

def format_volume(volume):
    if volume >= 1_000_000_000: return f"{volume/1_000_000_000:.2f}B"
    if volume >= 1_000_000: return f"{volume/1_000_000:.2f}M"
    return f"{volume:,.0f}"

def get_sentiment_color(sentiment):
    sentiment = str(sentiment).lower()
    if 'bullish' in sentiment or 'positive' in sentiment: return 'green'
    if 'bearish' in sentiment or 'negative' in sentiment: return 'red'
    return 'gray'

# API functions
def fetch_alpha_vantage_news(asset_category, asset_symbol, start_date=None, end_date=None):
    topics = {
        "crypto": "blockchain,cryptocurrency",
        "commodity": "economy_fiscal,economy_monetary",
        "stock": "earnings,ipo,mergers_and_acquisitions,financial_markets",
        "forex": "forex,economy_fiscal,economy_monetary"
    }.get(asset_category, "financial_markets")
    
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "tickers": get_alpha_symbol(asset_category, asset_symbol),
        "time_from": (start_date or (datetime.now() - timedelta(days=7))).strftime("%Y%m%dT0000"),
        "limit": 50,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    if end_date: params["time_to"] = end_date.strftime("%Y%m%dT2359")
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json().get("feed", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def fetch_alpha_vantage_data(asset_category, asset_symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    alpha_symbol = get_alpha_symbol(asset_category, asset_symbol)
    params = {"apikey": ALPHA_VANTAGE_API_KEY, "outputsize": outputsize}
    
    if asset_category == "crypto":
        params.update({"function": "DIGITAL_CURRENCY_DAILY", "symbol": alpha_symbol, "market": "USD"})
    elif asset_category == "forex":
        from_symbol, to_symbol = alpha_symbol.split("/")
        params.update({"function": "FX_DAILY", "from_symbol": from_symbol, "to_symbol": to_symbol})
    else:
        params.update({"function": function, "symbol": alpha_symbol})
    
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"API error: {data['Error Message']}")
            return pd.DataFrame()
            
        time_series_key = next((k for k in ["Time Series (Digital Currency Daily)", "Time Series FX (Daily)", 
                                          "Time Series (Daily)", "Weekly Time Series", "Monthly Time Series"] 
                              if k in data), None)
        if not time_series_key: return pd.DataFrame()
            
        df = pd.DataFrame(data[time_series_key]).T
        df.index = pd.to_datetime(df.index)
        rename_cols = {
            "1a. open (USD)": "Open", "2a. high (USD)": "High", "3a. low (USD)": "Low", "4a. close (USD)": "Close",
            "1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
        }
        df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})
        return df.apply(pd.to_numeric, errors="coerce").sort_index()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# RAG and AI functions
def create_rag_system(asset_category, asset_symbol, news_articles):
    if not news_articles: return None
    
    temp_dir = os.path.join(DATA_DIR, f"{asset_category}_{asset_symbol}")
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, article in enumerate(news_articles):
        ticker_symbol = get_alpha_symbol(asset_category, asset_symbol)
        ticker_sentiment = next((f"Score: {s.get('ticker_sentiment_score', 'N/A')}, Label: {s.get('ticker_sentiment_label', 'N/A')}" 
                               for s in article.get("ticker_sentiment", []) if s.get("ticker") == ticker_symbol), "N/A")
        
        content = f"""TITLE: {article.get('title', 'No title')}
DATE: {article.get('time_published', 'No date')}
SOURCE: {article.get('source', 'Unknown')}
AUTHORS: {', '.join(article.get('authors', [])) or 'Unknown'}
URL: {article.get('url', 'No URL')}
OVERALL SENTIMENT SCORE: {article.get('overall_sentiment_score', 'N/A')}
OVERALL SENTIMENT LABEL: {article.get('overall_sentiment_label', 'N/A')}
TICKER SPECIFIC SENTIMENT ({TICKERS[asset_category][asset_symbol]['name']}): {ticker_sentiment}

SUMMARY: {article.get('summary', 'No content')}
"""
        with open(os.path.join(temp_dir, f"article_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for file in os.listdir(temp_dir):
        loader = TextLoader(os.path.join(temp_dir, file), encoding="utf-8")
        documents.extend(text_splitter.split_documents(loader.load()))
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    return FAISS.from_documents(documents, embeddings)

def openrouter_llm_call(prompt, max_tokens=1000):
    try:
        completion = openrouter_client.chat.completions.create(
            extra_headers={"HTTP-Referer": "https://financial-insights-hub.streamlit.app", "X-Title": "Financial Insights Hub"},
            model="nvidia/llama-3.1-nemotron-70b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"API error: {e}")
        return f"Error: {str(e)}"

def summarize_daily_news(asset_category, asset_symbol, news_articles, date_str):
    date_articles = [a for a in news_articles if date_str in a.get("time_published", "")]
    if not date_articles: return "No articles found."
    
    content = "\n---\n\n".join(
        f"TITLE: {a.get('title', 'No title')}\nSOURCE: {a.get('source', 'Unknown')}\n"
        f"Overall sentiment: {a.get('overall_sentiment_label', 'N/A')}\n\nSUMMARY: {a.get('summary', '')}"
        for a in date_articles[:5]
    )
    
    prompt = f"""You are a financial analyst. Below are news articles about {TICKERS[asset_category][asset_symbol]['name']} from {date_str}:
    
{content}
    
Please provide:
1. A concise summary of key developments
2. Overall sentiment (positive, negative, neutral)
3. Potential market impact
4. Key factors driving sentiment"""
    
    return openrouter_llm_call(prompt)

def calculate_sentiment_score(asset_category, asset_symbol, news_articles):
    if not news_articles: return 0
    
    ticker_symbol = get_alpha_symbol(asset_category, asset_symbol)
    scores = [float(s["ticker_sentiment_score"]) for a in news_articles if "ticker_sentiment" in a 
             for s in a["ticker_sentiment"] if s.get("ticker") == ticker_symbol and "ticker_sentiment_score" in s]
    
    if not scores:
        scores = [float(a["overall_sentiment_score"]) for a in news_articles if "overall_sentiment_score" in a]
    
    return sum(scores)/len(scores) if scores else 0

def ask_rag(question, vectorstore, asset_category, asset_symbol):
    if not vectorstore: return "No data available."
    
    context = "\n\n".join(d.page_content for d in vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(question))
    prompt = f"""You are an AI assistant for {TICKERS[asset_category][asset_symbol]['name']}. Use this context:
    
{context}
    
QUESTION: {question}
    
ANSWER:"""
    return openrouter_llm_call(prompt)

def predict_price(asset_category, asset_symbol, data, sentiment_score):
    if data.empty or len(data) < 50: return "Insufficient data."
    
    df = data.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    latest = df.iloc[-1]
    
    prompt = f"""Predict tomorrow's price for {TICKERS[asset_category][asset_symbol]['name']}:
    
Current price: ${latest['Close']:.2f}
5-day change: {((df['Close'].iloc[-1]/df['Close'].iloc[-5])-1)*100:.2f}%
RSI (14): {latest.get('RSI_14', np.nan):.2f}
MACD: {latest.get('MACD_12_26_9', np.nan):.4f}
MACD Signal: {latest.get('MACDs_12_26_9', np.nan):.4f}
Bollinger Band Upper: ${latest.get('BBU_20_2.0', np.nan):.2f}
Bollinger Band Lower: ${latest.get('BBL_20_2.0', np.nan):.2f}
News Sentiment Score: {sentiment_score:.2f}
    
Provide:
1. Specific price prediction
2. Reasoning
3. Confidence level
4. Key factors"""
    return openrouter_llm_call(prompt, max_tokens=800)

# Display functions
def display_home_page():
    st.title("🌟 Financial Insights Hub")
    st.markdown("""### Welcome to your comprehensive financial analysis platform!
    This application provides real-time financial insights, news analysis, and technical indicators.
    #### 🔑 Key Features:
    - **Multi-Asset Coverage**: Analyze cryptocurrencies, stocks, indices, forex, and commodities
    - **News Sentiment Analysis**: Track market sentiment from financial news
    - **Technical Indicators**: View key technical analysis metrics
    - **AI-Powered Insights**: Ask questions about market trends
    - **Price Predictions**: AI-generated price forecasts""")
    
    fig = px.bar(x=["Crypto", "Stocks", "Forex", "Commodities", "Indices"],
                y=[0.45, 0.12, -0.25, 0.32, -0.15],
                color=[0.45, 0.12, -0.25, 0.32, -0.15],
                color_continuous_scale=["red", "gray", "green"],
                range_color=[-1, 1],
                labels={"x": "Asset Category", "y": "Market Sentiment"},
                title="Current Market Sentiment by Asset Class")
    st.plotly_chart(fig, use_container_width=True)

def display_overview(asset_category, asset_symbol, news_articles, data):
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not data.empty:
            fig = go.Figure(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                         low=data['Low'], close=data['Close'], name='Price'))
            fig.update_layout(title=f"{TICKERS[asset_category][asset_symbol]['name']} Price Movement",
                            xaxis_title="Date", yaxis_title="Price (USD)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else: st.error("No financial data available.")
    
    with col2:
        if not data.empty:
            latest, prev = data.iloc[-1], data.iloc[-2] if len(data) > 1 else data.iloc[-1]
            change, change_pct = latest["Close"] - prev["Close"], (latest["Close"]/prev["Close"]-1)*100
            st.metric("Current Price", f"${latest['Close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
            
            if "Volume" in data.columns and not np.isnan(latest["Volume"]):
                st.metric("Volume", format_volume(latest["Volume"]))
            
            try:
                year_data = data.last('365D')
                st.metric("52-Week Range", f"${year_data['Low'].min():.2f} - ${year_data['High'].max():.2f}")
            except: pass
        
        st.subheader("News Sentiment")
        if news_articles:
            avg_sentiment = calculate_sentiment_score(asset_category, asset_symbol, news_articles)
            fig = go.Figure(go.Indicator(mode="gauge+number", value=avg_sentiment, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "News Sentiment"}, gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "darkblue"},
                'steps': [{'range': [-1, -0.33], 'color': "red"}, {'range': [-0.33, 0.33], 'color': "gray"}, 
                         {'range': [0.33, 1], 'color': "green"}]}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Recent Headlines")
            for article in news_articles[:3]:
                sentiment = next((ts.get("ticker_sentiment_label", "N/A") for ts in article.get("ticker_sentiment", []) 
                                if ts.get("ticker") == get_alpha_symbol(asset_category, asset_symbol)), 
                               article.get("overall_sentiment_label", "N/A"))
                color = get_sentiment_color(sentiment)
                st.markdown(f":{color}_circle: **{article.get('title', 'No title')}** - *{sentiment}*")
    
    st.subheader("AI Price Prediction")
    if asset_symbol in st.session_state.predictions:
        st.markdown(st.session_state.predictions[asset_symbol])
    elif not data.empty and news_articles:
        prediction = predict_price(asset_category, asset_symbol, data, 
                                 calculate_sentiment_score(asset_category, asset_symbol, news_articles))
        st.session_state.predictions[asset_symbol] = prediction
        st.markdown(prediction)

def display_news_analysis(asset_category, asset_symbol, news_articles, start_date, end_date):
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} News Analysis")
    if not news_articles: return st.warning("No news articles available.")
    
    news_by_date = {}
    for article in news_articles:
        date_str = article.get("time_published", "")[:10]
        if date_str: news_by_date.setdefault(date_str, []).append(article)
    
    dates = sorted(news_by_date.keys(), reverse=True)
    st.subheader("News Volume by Date")
    fig = px.bar(x=list(news_by_date.keys()), y=[len(v) for v in news_by_date.values()],
                labels={"x": "Date", "y": "Number of Articles"}, title="News Volume")
    st.plotly_chart(fig, use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs(["📝 Daily Summaries", "📰 All Articles", "❓ Ask about News"])
    
    with tab1:
        st.subheader("AI-Generated Daily News Summaries")
        for date in dates:
            with st.expander(f"Summary for {date}"):
                key = f"{asset_symbol}_{date}"
                if key in st.session_state.daily_summaries:
                    st.markdown(st.session_state.daily_summaries[key])
                else:
                    with st.spinner(f"Generating summary for {date}..."):
                        summary = summarize_daily_news(asset_category, asset_symbol, news_by_date[date], date)
                        st.session_state.daily_summaries[key] = summary
                        st.markdown(summary)
    
    with tab2:
        st.subheader("All News Articles")
        articles_data = [{
            "Date": a.get("time_published", "")[:10],
            "Source": a.get("source", "Unknown"),
            "Title": a.get("title", "No title"),
            "Sentiment": next((ts.get("ticker_sentiment_label", "N/A") for ts in a.get("ticker_sentiment", []) 
                             if ts.get("ticker") == get_alpha_symbol(asset_category, asset_symbol)),
            "Sentiment Score": next((float(ts.get("ticker_sentiment_score", 0)) for ts in a.get("ticker_sentiment", []) 
                                   if ts.get("ticker") == get_alpha_symbol(asset_category, asset_symbol)),
            "URL": a.get("url", "#")
        } for a in news_articles]
        
        if articles_data:
            df = pd.DataFrame(articles_data)
            st.dataframe(df.style.applymap(lambda x: f"background-color: rgba(0, 128, 0, 0.2)" if 'bullish' in str(x).lower() or 'positive' in str(x).lower() else 
                                         (f"background-color: rgba(255, 0, 0, 0.2)" if 'bearish' in str(x).lower() or 'negative' in str(x).lower() else ''),
                        subset=['Sentiment']),
                        column_config={"URL": st.column_config.LinkColumn("Link")},
                        hide_index=True, use_container_width=True)
    
    with tab3:
        st.subheader("Ask Questions About the News")
        key = f"{asset_category}_{asset_symbol}"
        if key not in st.session_state.rag_index:
            with st.spinner("Building knowledge base..."):
                st.session_state.rag_index[key] = create_rag_system(asset_category, asset_symbol, news_articles)
        
        user_question = st.text_input("Ask a question:", key=f"question_{key}")
        if user_question:
            st.session_state.chat_history.setdefault(key, []).append({"role": "user", "content": user_question})
            with st.spinner("Analyzing..."):
                answer = ask_rag(user_question, st.session_state.rag_index[key], asset_category, asset_symbol)
                st.session_state.chat_history[key].append({"role": "assistant", "content": answer})
        
        if key in st.session_state.chat_history:
            st.subheader("Conversation History")
            for msg in st.session_state.chat_history[key]:
                st.markdown(f"**{'You' if msg['role'] == 'user' else 'Assistant'}:** {msg['content']}\n---")

def display_technical_analysis(asset_category, asset_symbol, data):
    st.header(f"{TICKERS[asset_category][asset_symbol]['name']} Technical Analysis")
    if data.empty: return st.error("No financial data available.")
    
    df = data.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    latest = df.iloc[-1]
    
    tab1, tab2, tab3 = st.tabs(["📊 Price & Moving Averages", "🔍 Oscillators", "📈 Advanced Analysis"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        for length, color in [(20, 'blue'), (50, 'orange'), (200, 'red')]:
            fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{length}'], line=dict(color=color, width=1), name=f'SMA {length}'))
        fig.update_layout(title=f"{TICKERS[asset_category][asset_symbol]['name']} Price with Moving Averages",
                         xaxis_title="Date", yaxis_title="Price (USD)", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Moving Average Analysis")
        cols = st.columns(3)
        for col, length in zip(cols, [20, 50, 200]):
            sma = latest[f'SMA_{length}']
            col.metric(f"SMA {length}", f"${sma:.2f}", 
                      f"{((latest['Close']/sma)-1)*100:.2f}%" if not np.isnan(sma) else None)
        
        if not np.isnan(latest['SMA_50']) and not np.isnan(latest['SMA_200']):
            if latest['SMA_50'] > latest['SMA_200']:
                previous = df.iloc[-10:]
                crossover = any(previous['SMA_50'].iloc[i-1] <= previous['SMA_200'].iloc[i-1] and 
                              previous['SMA_50'].iloc[i] > previous['SMA_200'].iloc[i] for i in range(1, len(previous)))
                if crossover: st.success("🚀 Golden Cross detected! Bullish signal.")
                else: st.info("📈 Bullish Trend: 50-day SMA above 200-day SMA.")
            else:
                previous = df.iloc[-10:]
                crossover = any(previous['SMA_50'].iloc[i-1] >= previous['SMA_200'].iloc[i-1] and 
                              previous['SMA_50'].iloc[i] < previous['SMA_200'].iloc[i] for i in range(1, len(previous)))
                if crossover: st.error("⚠️ Death Cross detected! Bearish signal.")
                else: st.warning("📉 Bearish Trend: 50-day SMA below 200-day SMA.")
    
    with tab2:
        fig = go.Figure(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=2)))
        for y, color in [(70, "red"), (30, "green"), (50, "gray")]: fig.add_hline(y=y, line_dash="dash", line_color=color)
        fig.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI Value", height=400, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)
        
        if not np.isnan(latest['RSI_14']):
            st.metric("Current RSI (14)", f"{latest['RSI_14']:.2f}")
            if latest['RSI_14'] > 70: st.warning("🔥 Overbought")
            elif latest['RSI_14'] < 30: st.success("❄️ Oversold")
            else: st.info("🔄 Neutral RSI Zone")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='blue', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='red', width=1.5)), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], marker_color=['green' if x >=0 else 'red' for x in df['MACDh_12_26_9']]), row=2, col=1)
        fig.update_layout(title="MACD (12,26,9)", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        if not np.isnan(latest['MACD_12_26_9']) and not np.isnan(latest['MACDs_12_26_9']):
            cols = st.columns(3)
            cols[0].metric("MACD", f"{latest['MACD_12_26_9']:.4f}")
            cols[1].metric("Signal", f"{latest['MACDs_12_26_9']:.4f}")
            cols[2].metric("Histogram", f"{latest['MACDh_12_26_9']:.4f}")
            
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                st.success("🚀 Strong Bullish Signal" if latest['MACD_12_26_9'] > 0 else "📈 Bullish Signal")
            else:
                st.error("📉 Strong Bearish Signal" if latest['MACD_12_26_9'] < 0 else "⚠️ Bearish Signal")
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        for band, color in [('BBU_20_2.0', 'gray'), ('BBM_20_2.0', 'blue'), ('BBL_20_2.0', 'gray')]:
            fig.add_trace(go.Scatter(x=df.index, y=df[band], line=dict(color=color, width=1), name=band))
        fig.update_layout(title="Bollinger Bands (20,2)", xaxis_title="Date", yaxis_title="Price", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        if not np.isnan(latest['BBU_20_2.0']) and not np.isnan(latest['BBL_20_2.0']):
            bandwidth = (latest['BBU_20_2.0'] - latest['BBL_20_2.0']) / latest['BBM_20_2.0'] * 100
            percent_b = (latest['Close'] - latest['BBL_20_2.0']) / (latest['BBU_20_2.0'] - latest['BBL_20_2.0']) * 100
            cols = st.columns(2)
            cols[0].metric("Bandwidth", f"{bandwidth:.2f}%")
            cols[1].metric("%B", f"{percent_b:.2f}%")
            
            if latest['Close'] > latest['BBU_20_2.0']: st.warning("🚨 Overbought")
            elif latest['Close'] < latest['BBL_20_2.0']: st.success("⭐ Oversold")
            else: st.info("➡️ Within Bands")
            
            if bandwidth < 10: st.info("📊 Low Volatility")
            elif bandwidth > 30: st.warning("📊 High Volatility")

# Main app
def main():
    st.sidebar.title("Financial Insights Hub")
    selected_page = st.sidebar.radio("Navigation", ["Home", "Asset Analysis"])
    
    if selected_page == "Home":
        display_home_page()
    else:
        st.sidebar.header("Asset Selection")
        asset_category = st.sidebar.selectbox("Asset Category", list(TICKERS.keys()), format_func=lambda x: x.capitalize())
        asset_symbol = st.sidebar.selectbox(f"Select {asset_category.capitalize()}", list(TICKERS[asset_category].keys()),
                                          format_func=lambda x: f"{x} - {TICKERS[asset_category][x]['name']}")
        
        today = datetime.now().date()
        start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=30), max_value=today)
        end_date = st.sidebar.date_input("End Date", value=today, max_value=today, min_value=start_date)
        
        if st.sidebar.button("Refresh Data"):
            st.session_state.pop(f"{asset_category}_{asset_symbol}", None)
            st.session_state.pop(f"{asset_category}_{asset_symbol}_news", None)
        
        data_key = f"{asset_category}_{asset_symbol}"
        news_key = f"{asset_category}_{asset_symbol}_news"
        
        if data_key not in st.session_state.last_updated:
            with st.spinner("Fetching financial data..."):
                st.session_state[data_key] = fetch_alpha_vantage_data(asset_category, asset_symbol)
                st.session_state.last_updated[data_key] = datetime.now()
        
        if news_key not in st.session_state.last_updated:
            with st.spinner("Fetching news data..."):
                news_articles = fetch_alpha_vantage_news(asset_category, asset_symbol, start_date, end_date)
                st.session_state[news_key] = news_articles
                st.session_state.last_updated[news_key] = datetime.now()
                st.session_state.news_data[news_key] = news_articles
        
        data = st.session_state.get(data_key, pd.DataFrame())
        news_articles = st.session_state.get(news_key, [])
        
        if data_key in st.session_state.last_updated:
            st.sidebar.caption(f"Last updated: {st.session_state.last_updated[data_key].strftime('%Y-%m-%d %H:%M:%S')}")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "News Analysis", "Technical Analysis"])
        with tab1: display_overview(asset_category, asset_symbol, news_articles, data)
        with tab2: display_news_analysis(asset_category, asset_symbol, news_articles, start_date, end_date)
        with tab3: display_technical_analysis(asset_category, asset_symbol, data)

if __name__ == "__main__":
    main()