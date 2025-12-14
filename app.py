# ==========================================
# 區塊 1: 匯入工具箱
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
import twstock
import requests
from bs4 import BeautifulSoup
import time
import google.generativeai as genai
from duckduckgo_search import DDGS  # 搜尋新聞用
import requests_cache  # 為了繞過 Yahoo 擋 IP 機制

# ==========================================
# 區塊 2: 網頁基礎設定
# ==========================================
st.set_page_config(page_title="AI 智能台股分析 v3.1", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統 (v3.1)")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)**、**蒙地卡羅模擬 (Risk)** 與 **Generative AI (多源輿情)** 的全方位決策系統。
> **技術架構**：Python ETL + Gemini LLM + Monte Carlo Simulation + PTT Crawler
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
        st.caption("提示：部署到 Streamlit Cloud 後可設定 Secrets 隱藏此欄位")

# ==========================================
# 區塊 4: AI 模型選擇器
# ==========================================
selected_model_name = "gemini-1.5-flash"
if api_key:
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        
        target_models = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-1.5-flash-8b'
        ]
        
        try:
            api_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            api_models = []
            
        all_options = list(set(target_models + api_models))
        all_options.sort()
        
        priorities = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for p in reversed(priorities):
            if p in all_options:
                all_options.remove(p)
                all_options.insert(0, p)

        selected_model_name = st.sidebar.selectbox("選擇推論模型 (Model)", all_options, index=0)
        
        if "exp" in selected_model_name:
            st.sidebar.success(f"🚀 已啟用最新實驗版: {selected_model_name}")
        elif "flash" in selected_model_name:
            st.sidebar.info(f"⚡ 已啟用高速推論模式")
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

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
    keys_to_clear = ['run_mc', 'mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

if ticker.isdigit():
    ticker = f"{ticker}.TW"

# ==========================================
# 區塊 6: 核心功能函數定義
# ==========================================

# 🟢 1. DuckDuckGo 新聞搜尋
def fetch_news_ddg(keywords, limit=5):
    try:
        results = DDGS().news(keywords=keywords, region="wt-wt", safesearch="off", max_results=limit)
        news_list = []
        if results:
            for item in results:
                news_list.append({'title': item['title'], 'link': item['url']})
        return news_list
    except Exception as e:
        print(f"DDG Search Error: {e}")
        return []

# 🟢 2. PTT 爬蟲 (含 Referer)
@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, code, limit=5, retries=3):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={code}+OR+{keyword}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Cookie': 'over18=1',
        'Referer': 'https://www.ptt.cc/bbs/Stock/index.html'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                titles = soup.find_all('div', class_='title')
                result = []
                for t in titles[:limit]:
                    a_tag = t.find('a')
                    if a_tag:
                        result.append(a_tag.text.strip())
                return result
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
                continue
    return []

@st.cache_data
def calculate_metrics(df):
    close = df['Close'].ffill()
    log_returns = np.log(close / close.shift(1))
    u = log_returns.mean()
    var = log_returns.var()
    daily_volatility = log_returns.std()
    drift = u - (0.5 * var)
    annual_volatility = daily_volatility * np.sqrt(252)
    return log_returns, daily_volatility, drift, annual_volatility

@st.cache_data(ttl=600)
def run_monte_carlo(last_price, drift, daily_volatility, sim_days, n_simulations):
    all_paths = []
    for i in range(n_simulations):
        daily_shocks = drift + daily_volatility * np.random.normal(0, 1, sim_days)
        price_paths = [last_price]
        for shock in daily_shocks:
            price_paths.append(price_paths[-1] * np.exp(shock))
        all_paths.append(price_paths)
    
    fig_mc = go.Figure()
    x_axis = list(range(sim_days + 1))
    
    # 繪製路徑 (透明)
    for path in all_paths[:100]:
        fig_mc.add_trace(go.Scatter(x=x_axis, y=path, mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)', width=1), showlegend=False, hovertemplate="第%{x}天: $%{y:.2f}"))
    
    # 繪製平均線
    avg_path = np.mean(all_paths, axis=0)
    fig_mc.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', line=dict(color='orange', width=3), name='平均預期'))
    
    fig_mc.update_layout(title=f"未來 {sim_days} 天股價模擬 ({n_simulations} 次運算)", xaxis_title="天數", yaxis_title="股價", height=500)
    
    final_prices = [p[-1] for p in all_paths]
    expected_return = (np.mean(final_prices) - last_price) / last_price
    var_95_price = np.percentile(final_prices, 5)
    loss_at_risk = (last_price - var_95_price) / last_price
    
    return fig_mc, expected_return, loss_at_risk

# ==========================================
# 區塊 7: 主程式邏輯
# ==========================================
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False
if 'run_mc' not in st.session_state:
    st.session_state.run_mc = False

def start_analysis_callback():
    st.session_state['analysis_started'] = True

st.button("🚀 啟動全方位分析", on_click=start_analysis_callback)

tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬 (Risk Lab)"])

# --- Tab 2: 蒙地卡羅說明 ---
with tab2:
    st.header("🎲 蒙地卡羅風險模擬")
    with st.expander("📖 模擬原理說明", expanded=True):
        st.info("基於幾何布朗運動 (GBM)，模擬 500+ 條平行時空的股價走勢，計算 95% 信賴區間下的最大可能虧損 (VaR)。")

# --- Tab 1: 提示 ---
if not st.session_state['analysis_started']:
    with tab1:
        st.info("👈 請在左側設定參數，並點擊上方「🚀 啟動全方位分析」按鈕開始。")

# 執行邏輯
if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。")
        st.stop()
        
    # --- 🟢 ETL (核心修正：防擋 + 容錯) ---
    try:
        # 設定 requests_cache session 來偽裝瀏覽器
        session = requests_cache.CachedSession('yfinance.cache')
        session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        
        try:
            stock_obj = yf.Ticker(ticker, session=session)
        except:
            stock_obj = yf.Ticker(ticker)

        # 改用 period='1y' 避開週末日期問題
        df = stock_obj.history(period="1y")
        
        # 自動切換 .TWO
        if df.empty and ".TW" in ticker:
            alt_ticker = ticker.replace(".TW", ".TWO")
            st.warning(f"⚠️ 找不到 {ticker}，嘗試切換為 {alt_ticker}...")
            stock_obj = yf.Ticker(alt_ticker, session=session)
            df = stock_obj.history(period="1y")
            ticker = alt_ticker

        # 本地篩選天數
        if not df.empty:
            filter_date = df.index[-1] - timedelta(days=days)
            df = df[df.index >= filter_date]

        # 獲取 Beta (容錯)
        beta = 1.0
        try:
            info = stock_obj.info
            val = info.get('beta')
            if val is not None: beta = val
        except:
            pass
        
        if df.empty or len(df) < 30:
            st.error(f"❌ 無法取得 {ticker} 資料。請確認代號或網路狀態。")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (2 * df['STD'])
        df['Lower'] = df['MA20'] - (2 * df['STD'])
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]

    except Exception as e:
        st.error(f"數據處理錯誤: {e}")
        st.stop()

    # --- 顯示 Tab 1 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta", f"{beta:.2f}")

        # 圖表繪製
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', showlegend=False, name='Bollinger'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        fig.add_hrect(y0=30, y1=70, row=2, col=1, fillcolor="gray", opacity=0.1, line_width=0)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        st.markdown("---")
        col_news, col_ai = st.columns([1, 1])
        
        news_text_for_ai = ""
        ptt_text_for_ai = ""
        
        with col_news:
            st.subheader("📰 多源輿情偵測")
            with st.spinner("🔍 搜尋最新新聞中..."):
                news_items = fetch_news_ddg(f"{stock_name} 股票", limit=4)
                if news_items:
                    for item in news_items:
                        st.write(f"- [{item['title']}]({item['link']})")
                        news_text_for_ai += f"{item['title']}\n"
                else:
                    st.caption("無即時新聞，將依賴歷史數據。")
            
            st.markdown("**PTT 股版散戶熱議**")
            code_num = ticker.replace('.TW', '').replace('.TWO', '')
            ptt_titles = fetch_ptt_sentiment(stock_name, code_num, limit=3)
            if ptt_titles:
                for t in ptt_titles:
                    st.write(f"- 💬 {t}")
                    ptt_text_for_ai += f"{t}\n"
            else:
                st.caption("無近期相關討論")
                
        with col_ai:
            st.subheader("🤖 Gemini 雙軌決策報告")
            with st.spinner("AI 正在推論 (Chain of Thought)..."):
                try:
                    model = genai.GenerativeModel(selected_model_name, generation_config=genai.types.GenerationConfig(temperature=0.2))
                    today_str = datetime.now().strftime("%Y年%m月%d日")
                    
                    prompt = f"""
                    你是一位專業量化交易員。今天是 {today_str}。
                    目標股票：{stock_name} ({ticker})，收盤價：{last_close}。
                    
                    ### 輸入數據
                    1. **技術指標**：RSI={df['RSI'].iloc[-1]:.2f}, MA20={df['MA20'].iloc[-1]:.2f}, Beta={beta:.2f}
                    2. **主流新聞**：\n{news_text_for_ai}
                    3. **社群論壇(PTT)**：\n{ptt_text_for_ai}
                    
                    ### 任務
                    請以 **純 JSON** 格式輸出分析結果。不要包含 Markdown 標記。
                    格式：
                    {{
                        "sentiment_weight": 70,
                        "weight_reason": "簡短理由...",
                        "chart_data": {{ "target_price": 0, "high_price": 0, "low_price": 0 }},
                        "analysis_report": "使用 Markdown 撰寫的詳細報告..."
                    }}
                    """
                    response = model.generate_content(prompt)
                    
                    # 🟢 JSON 強健解析
                    raw_text = response.text
                    json_start = raw_text.find('{')
                    json_end = raw_text.rfind('}')
                    
                    if json_start != -1 and json_end != -1:
                        json_str = raw_text[json_start : json_end+1]
                        ai_data = json.loads(json_str)
                        
                        w = ai_data.get('sentiment_weight', 50)
                        st.info(f"⚖️ 消息權重: {w}% | 技術權重: {100-w}%")
                        st.progress(w/100)
                        
                        if 'analysis_report' in ai_data:
                            st.markdown(ai_data['analysis_report'])
                            
                        if 'chart_data' in ai_data:
                            c = ai_data['chart_data']
                            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=10)
                            next_dt = future_dates[0] if len(future_dates)>0 else last_date
                            
                            fig.add_trace(go.Scatter(x=[last_date, next_dt], y=[last_close, c.get('high_price', last_close)], mode='lines+markers', line=dict(color='red', dash='dot'), name='樂觀'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=[last_date, next_dt], y=[last_close, c.get('low_price', last_close)], mode='lines+markers', line=dict(color='green', dash='dot'), name='悲觀'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=[last_date, next_dt], y=[last_close, c.get('target_price', last_close)], mode='lines+markers', line=dict(color='orange', width=4), name='目標'), row=1, col=1)
                    else:
                        st.error("AI 回傳格式異常，請重試。")

                except Exception as e:
                    st.error(f"AI 分析失敗: {e}")
        
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: 蒙地卡羅運算 ---
    with tab2:
        st.divider()
        mc_col1, mc_col2 = st.columns([1, 3])
        
        try:
            log_returns, daily_volatility, drift, annual_volatility = calculate_metrics(df)
        except:
            st.stop()

        with mc_col1:
            st.subheader("參數設定")
            sim_days = st.slider("模擬天數", 30, 365, 90)
            n_simulations = st.slider("模擬次數", 100, 1000, 500)
            initial_investment = st.number_input("投資金額", value=100000, step=10000)
            st.metric("年化波動率", f"{annual_volatility*100:.2f}%")
            st.metric("日均漂移率", f"{drift*100:.4f}%")
        
        with mc_col2:
            col_btn, col_clear = st.columns([1, 4])
            with col_btn:
                if st.button("🎲 開始模擬", type="primary", use_container_width=True):
                    with st.spinner("計算中..."):
                        fig_mc, expected_return, loss_at_risk = run_monte_carlo(last_close, drift, daily_volatility, sim_days, n_simulations)
                        st.session_state.mc_fig = fig_mc
                        st.session_state.mc_return = expected_return
                        st.session_state.mc_risk = loss_at_risk
                        st.session_state.mc_asset = initial_investment * (1 - loss_at_risk)
                        st.session_state.run_mc = True
            
            if st.session_state.run_mc and 'mc_fig' in st.session_state:
                st.plotly_chart(st.session_state.mc_fig, use_container_width=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("預期報酬", f"{st.session_state.mc_return*100:.2f}%")
                r2.metric("95% VaR", f"-{st.session_state.mc_risk*100:.2f}%")
                r3.metric("剩餘資產", f"${st.session_state.mc_asset:,.0f}")
            
            with col_clear:
                if st.session_state.run_mc:
                    if st.button("清除結果"):
                        st.session_state.run_mc = False
                        st.rerun()
