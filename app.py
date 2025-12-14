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

# ==========================================
# 區塊 2: 網頁基礎設定
# ==========================================
st.set_page_config(page_title="AI 智能台股分析 v3.2", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統 (v3.2)")
st.markdown("""
> **版本更新 (v3.2)**：新增「雙重數據源」機制 (Yahoo + TWSE)，解決雲端 IP 被封鎖導致無法抓取股價的問題。
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
# 區塊 4: AI 模型設定
# ==========================================
selected_model_name = "gemini-1.5-flash"
if api_key:
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        # 簡化模型選擇，優先使用穩定快速的模型
        selected_model_name = st.sidebar.selectbox("選擇模型", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"], index=0)
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
    
    # 清除舊的模擬結果
    keys_to_clear = ['run_mc', 'mc_fig']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

if ticker.isdigit():
    ticker = f"{ticker}.TW"

# ==========================================
# 區塊 6: 核心功能函數
# ==========================================

@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, limit=3):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'Cookie': 'over18=1'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            titles = soup.find_all('div', class_='title')
            return [t.find('a').text.strip() for t in titles if t.find('a')][:limit]
    except:
        pass
    return []

@st.cache_data
def calculate_metrics(df):
    close = df['Close'].ffill()
    log_returns = np.log(close / close.shift(1))
    daily_volatility = log_returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)
    drift = log_returns.mean() - (0.5 * log_returns.var())
    return drift, annual_volatility

# 🛡️ 強化的數據抓取函數 (核心修改)
def robust_fetch_stock(ticker_code, days_back):
    # 1. 嘗試 Yahoo Finance
    try:
        end = datetime.now()
        start = end - timedelta(days=days_back + 30)
        df = yf.Ticker(ticker_code).history(start=start, end=end)
        if not df.empty and len(df) > 10:
            return df, "Yahoo Finance"
    except:
        pass
    
    # 2. 備援：嘗試 TWStock (證交所)
    try:
        code_only = ticker_code.split('.')[0]
        stock = twstock.Stock(code_only)
        # 抓取近幾個月的資料
        data = stock.fetch_from(datetime.now().year, datetime.now().month - 3)
        
        if data:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            # 轉換欄位名稱與型態以符合 Yahoo 格式
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'capacity': 'Volume'})
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 篩選日期範圍
            start_filter = datetime.now() - timedelta(days=days_back + 30)
            df = df[df.index >= start_filter]
            return df, "TWSE (證交所)"
    except Exception as e:
        print(f"TWStock error: {e}")

    return pd.DataFrame(), "None"

# ==========================================
# 區塊 7: 主程式邏輯
# ==========================================
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False

if 'run_mc' not in st.session_state:
    st.session_state.run_mc = False

st.button("🚀 啟動全方位分析", on_click=lambda: st.session_state.update({'analysis_started': True}))
tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬"])

if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 請先輸入 Gemini API Key")
        st.stop()

    # --- ETL 資料處理 ---
    try:
        # 使用新的強固抓取函數
        df, source = robust_fetch_stock(ticker, days)
        
        if df.empty:
            st.error(f"❌ 無法取得 {ticker} 資料。請確認代號正確，或稍後再試。")
            st.stop()
        
        if source == "TWSE (證交所)":
            st.warning("⚠️ Yahoo 連線受阻，已自動切換至備用數據源 (TWSE)，載入速度可能稍慢。")
        else:
            st.toast(f"✅ 數據載入成功 ({source})")

        # 嘗試抓 Beta，抓不到就用預設值 1.0
        try:
            if source == "Yahoo Finance":
                beta = yf.Ticker(ticker).info.get('beta', 1.0)
            else:
                beta = 1.0 # TWStock 沒提供 Beta
            if beta is None: beta = 1.0
        except:
            beta = 1.0

        # 技術指標計算
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + (2 * df['STD'])
        df['Lower'] = df['MA20'] - (2 * df['STD'])
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]

    except Exception as e:
        st.error(f"資料運算錯誤: {e}")
        st.stop()

    # --- 分頁 1: AI 分析 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta", f"{beta:.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # AI 區塊
        col_news, col_ai = st.columns([1, 1])
        news_text = ""
        ptt_text = ""
        
        with col_news:
            st.subheader("📰 市場消息")
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                for item in googlenews.result()[:3]:
                    st.write(f"- [{item['title']}]({item['link']})")
                    news_text += f"{item['title']}\n"
            except:
                st.caption("新聞抓取受限")
            
            st.markdown("**PTT 熱議**")
            ptt_titles = fetch_ptt_sentiment(stock_name)
            for t in ptt_titles:
                st.write(f"- {t}")
                ptt_text += f"{t}\n"

        with col_ai:
            st.subheader("🤖 AI 決策建議")
            if st.button("開始 AI 分析 (需消耗 Token)"):
                with st.spinner("AI 思考中..."):
                    try:
                        model = genai.GenerativeModel(selected_model_name)
                        prompt = f"""
                        角色：專業操盤手。目標：{stock_name} ({ticker})，現價 {last_close}。
                        技術面：RSI={df['RSI'].iloc[-1]:.2f}, MA20={df['MA20'].iloc[-1]:.2f}。
                        消息面：\n{news_text}\nPTT:\n{ptt_text}
                        
                        請輸出 JSON 格式 (不要 Markdown):
                        {{
                            "sentiment_weight": 50,
                            "reason": "簡短理由",
                            "analysis": "Markdown 格式的完整分析",
                            "prediction": {{ "target": 0, "high": 0, "low": 0 }}
                        }}
                        """
                        response = model.generate_content(prompt)
                        # JSON 清洗與解析
                        clean_json = re.sub(r'```json|```', '', response.text).strip()
                        if '{' in clean_json: clean_json = clean_json[clean_json.find('{'):clean_json.rfind('}')+1]
                        
                        try:
                            ai_data = json.loads(clean_json)
                            w = ai_data.get('sentiment_weight', 50)
                            st.info(f"消息面權重: {w}% | {ai_data.get('reason')}")
                            st.markdown(ai_data.get('analysis'))
                            
                            pred = ai_data.get('prediction', {})
                            if pred.get('target', 0) > 0:
                                st.metric("AI 目標價", pred['target'], f"高點 {pred.get('high')} / 低點 {pred.get('low')}")
                        except:
                            st.error("AI 回傳格式錯誤，請重試")
                            st.write(response.text)
                    except Exception as e:
                        st.error(f"AI 連線錯誤: {e}")

    # --- 分頁 2: 蒙地卡羅 ---
    with tab2:
        st.subheader("🎲 風險模擬 (Monte Carlo)")
        c1, c2 = st.columns([1, 3])
        with c1:
            sim_days = st.slider("預測天數", 30, 365, 90)
            n_sims = st.slider("模擬次數", 100, 1000, 500)
            if st.button("開始模擬", type="primary"):
                st.session_state.run_mc = True
        
        if st.session_state.run_mc:
            drift, ann_vol = calculate_metrics(df)
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
            var95 = last_close - np.percentile(final_prices, 5)
            st.error(f"95% 風險值 (VaR): 若發生極端狀況，可能虧損 ${var95:.2f}")
