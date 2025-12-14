# ==========================================
# 區塊 1: 匯入工具箱
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from GoogleNews import GoogleNews
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import re
import twstock
import requests
from bs4 import BeautifulSoup
import time
import random

# ==========================================
# 區塊 2: 網頁基礎設定
# ==========================================
st.set_page_config(page_title="AI 智能台股分析 v3.3 (Force Start)", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統 (v3.3 暴力啟動版)")
st.markdown("""
> **版本特點**：新增 **Mock Data (模擬數據)** 機制。當 Yahoo 與證交所連線皆被封鎖時，系統將自動生成模擬股價，確保應用程式能順利啟動以供測試。
""")

# ==========================================
# 區塊 3: API 金鑰管理
# ==========================================
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except:
    pass

if not api_key:
    with st.sidebar.expander("🔐 API Key 設定", expanded=True):
        api_key = st.text_input("請輸入 Google Gemini API Key", type="password")

# ==========================================
# 區塊 4: AI 模型選擇器
# ==========================================
selected_model_name = "gemini-1.5-flash"
if api_key:
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        selected_model_name = st.sidebar.selectbox("選擇推論模型", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    except:
        pass

# ==========================================
# 區塊 5: 股票參數輸入
# ==========================================
st.sidebar.header("📊 股票參數")

def update_stock_name():
    input_val = st.session_state.ticker_input.strip()
    code = input_val.split('.')[0]
    if code in twstock.codes:
        st.session_state.stock_name_input = twstock.codes[code].name
    keys_to_clear = ['run_mc', 'mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

if ticker.isdigit():
    ticker = f"{ticker}.TW"

# ==========================================
# 區塊 6: 核心功能函數定義
# ==========================================

@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, limit=3):
    # 簡化版 PTT 爬蟲，若失敗直接回傳空值，避免卡住
    try:
        url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
        headers = {'User-Agent': 'Mozilla/5.0', 'Cookie': 'over18=1'}
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            return [t.find('a').text.strip() for t in soup.find_all('div', class_='title') if t.find('a')][:limit]
    except:
        pass
    return []

@st.cache_data
def calculate_metrics(df):
    close = df['Close'].ffill()
    log_returns = np.log(close / close.shift(1))
    drift = log_returns.mean() - (0.5 * log_returns.var())
    annual_volatility = log_returns.std() * np.sqrt(252)
    return log_returns, log_returns.std(), drift, annual_volatility

# 🚨🚨🚨 救命用的假資料生成器 🚨🚨🚨
def generate_mock_data(ticker_name, days_back):
    st.warning(f"⚠️ 檢測到網路封鎖！正在為 {ticker_name} 生成模擬數據以供測試...")
    
    # 建立日期索引
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 30)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # 隨機漫步生成股價
    np.random.seed(42) # 固定種子，讓每次跑起來一樣
    start_price = 1000 if "2330" in ticker_name else 100
    returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
    price_path = start_price * (1 + returns).cumprod()
    
    data = {
        'Open': price_path * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': price_path * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': price_path * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Close': price_path,
        'Volume': np.random.randint(1000, 50000, len(dates)) * 1000
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def robust_fetch_stock(ticker_code, days_back):
    # 1. Yahoo
    try:
        yf_ticker = ticker_code if ".TW" in ticker_code else f"{ticker_code}.TW"
        df = yf.Ticker(yf_ticker).history(period=f"{int(days_back*1.5)}d")
        if not df.empty: return df, "Yahoo Finance"
    except: pass
    
    # 2. TWStock
    try:
        clean = ticker_code.split('.')[0]
        stock = twstock.Stock(clean)
        data = stock.fetch_from(datetime.now().year, datetime.now().month - 3)
        if data:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'capacity': 'Volume'})
            for c in ['Close', 'Open', 'High', 'Low', 'Volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df, "TWSE"
    except: pass

    # 3. 模擬數據 (保底)
    return generate_mock_data(ticker_code, days_back), "⚠️ 模擬數據 (Mock Data)"

# ==========================================
# 區塊 7: 主程式邏輯
# ==========================================

# 初始化狀態
if 'analysis_started' not in st.session_state: st.session_state['analysis_started'] = False
if 'run_mc' not in st.session_state: st.session_state.run_mc = False

st.button("🚀 啟動全方位分析", on_click=lambda: st.session_state.update({'analysis_started': True}))
tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬"])

# 說明頁
with tab2:
    if not st.session_state['analysis_started']:
        st.info("👈 請點擊上方按鈕啟動。若網路受阻，系統將自動切換為模擬模式。")

if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key")
        st.stop()

    # --- ETL ---
    df, source = robust_fetch_stock(ticker, days)
    
    if "模擬" in source:
        st.error(f"無法連線至交易所，目前使用：{source}")
    else:
        st.toast(f"數據來源：{source}")

    # 技術指標
    try:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + 2*df['STD']
        df['Lower'] = df['MA20'] - 2*df['STD']
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        last_close = float(df['Close'].iloc[-1])
        beta = 1.2 # 模擬模式或失敗時的預設值
    except Exception as e:
        st.error(f"運算錯誤: {e}")
        st.stop()

    # --- Tab 1 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("收盤價", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta", f"{beta}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        col_news, col_ai = st.columns([1, 1])
        news_text, ptt_text = "", ""
        
        with col_news:
            st.subheader("📰 市場快訊")
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                for item in googlenews.result()[:3]:
                    st.write(f"- {item['title']}")
                    news_text += item['title']
            except: st.caption("新聞連線受阻")
            
            st.markdown("**PTT 討論**")
            ptt = fetch_ptt_sentiment(stock_name)
            if ptt: 
                for t in ptt: 
                    st.write(f"- {t}")
                    ptt_text += t
            else: st.caption("無資料")

        with col_ai:
            st.subheader("🤖 AI 決策")
            if st.button("生成分析報告"):
                with st.spinner("AI 運算中..."):
                    try:
                        model = genai.GenerativeModel(selected_model_name)
                        prompt = f"""
                        角色：量化分析師。目標：{stock_name}。現價：{last_close}。
                        技術面：RSI={df['RSI'].iloc[-1]:.2f}。
                        新聞：{news_text} PTT：{ptt_text}
                        
                        請輸出純 JSON:
                        {{
                            "sentiment_weight": 60,
                            "reason": "簡短理由",
                            "analysis": "詳細 Markdown 分析",
                            "prediction": {{ "target": 0 }}
                        }}
                        """
                        res = model.generate_content(prompt)
                        clean = re.sub(r'```json|```', '', res.text).strip()
                        if '{' in clean: clean = clean[clean.find('{'):clean.rfind('}')+1]
                        
                        data = json.loads(clean)
                        st.info(f"建議權重: {data.get('sentiment_weight')}% | {data.get('reason')}")
                        st.markdown(data.get('analysis'))
                    except Exception as e:
                        st.error(f"AI 錯誤: {e}")

    # --- Tab 2: Monte Carlo ---
    with tab2:
        st.subheader("🎲 風險模擬")
        c1, c2 = st.columns([1, 3])
        with c1:
            sim_days = st.slider("天數", 30, 365, 90)
            n_sims = st.slider("次數", 100, 1000, 500)
            if st.button("開始模擬"): st.session_state.run_mc = True
        
        if st.session_state.run_mc:
            ret, vol, drift, ann_vol = calculate_metrics(df)
            daily_vol = ann_vol / np.sqrt(252)
            
            paths = []
            for _ in range(n_sims):
                shocks = drift + daily_vol * np.random.normal(0, 1, sim_days)
                path = [last_close]
                for s in shocks: path.append(path[-1] * np.exp(s))
                paths.append(path)
            
            fig_mc = go.Figure()
            for p in paths[:100]:
                fig_mc.add_trace(go.Scatter(y=p, mode='lines', line=dict(width=1, color='rgba(100,100,255,0.1)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=np.mean(paths, axis=0), mode='lines', line=dict(width=3, color='orange'), name='平均路徑'))
            st.plotly_chart(fig_mc, use_container_width=True)
            
            final_prices = [p[-1] for p in paths]
            st.metric("預期報酬", f"{(np.mean(final_prices)-last_close)/last_close*100:.2f}%")
