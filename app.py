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

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 智能台股情緒量化分析系統", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)**、**蒙地卡羅模擬 (Risk)** 與 **Generative AI (多源輿情)** 的全方位決策系統。
> **技術架構**：Python ETL + Gemini LLM + Monte Carlo Simulation + PTT Crawler
""")

# --- 2. 智慧型 API Key 管理 ---
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

# --- 3. 進階模型選擇器 ---
# 預設改為你指定的 Gemma 模型
selected_model_name = "gemma-3n-e4b-it"

if api_key:
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        
        target_models = [
            'gemma-3n-e4b-it',              
            'gemini-2.5-pro-preview-03-25', 
            'gemini-1.5-pro',               
            'gemini-1.5-flash',             
            'gemini-pro'                    
        ]
        
        try:
            api_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            api_models = []
            
        all_options = list(set(target_models + api_models))
        all_options.sort()
        
        priorities = ['gemma-3n-e4b-it', 'gemini-2.5-pro-preview-03-25', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for p in reversed(priorities):
            if p in all_options:
                all_options.remove(p)
                all_options.insert(0, p)

        selected_model_name = st.sidebar.selectbox("選擇推論模型 (Model)", all_options, index=0)
        
        if "gemma" in selected_model_name:
            st.sidebar.warning(f"🧪 已啟用實驗性模型: {selected_model_name}")
        elif "preview" in selected_model_name:
            st.sidebar.success(f"🚀 已啟用最新預覽版: {selected_model_name}")
        elif "flash" in selected_model_name:
            st.sidebar.info(f"⚡ 已啟用高速推論模式")
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

# --- 4. 股票參數設定 ---
st.sidebar.header("📊 股票參數")

def update_stock_name():
    input_val = st.session_state.ticker_input.strip()
    code = input_val.split('.')[0]
    if code in twstock.codes:
        st.session_state.stock_name_input = twstock.codes[code].name

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

if ticker.isdigit():
    ticker = f"{ticker}.TW"

# --- 5. 核心功能函數 ---

@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, limit=5, retries=3):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'Cookie': 'over18=1'}
    
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
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    u = log_returns.mean()
    var = log_returns.var()
    daily_volatility = log_returns.std()
    drift = u - (0.5 * var)
    annual_volatility = daily_volatility * np.sqrt(252)
    return log_returns, daily_volatility, drift, annual_volatility

# --- 6. 主程式邏輯 ---

if st.button("🚀 啟動全方位分析"):
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。請在側邊欄輸入或檢查 Secrets 設定。")
        st.stop()

    tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬 (Risk Lab)"])

    # --- 共用資料處理 ---
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date, end=end_date)
        
        # 嘗試抓取 Beta 值
        try:
            stock_info = stock_obj.info
            if not stock_info:
                beta = 1.0
            else:
                beta = stock_info.get('beta')
                if beta is None: beta = 1.0
        except:
            beta = 1.0
        
        if df.empty:
            st.error(f"找不到 {ticker} 的股價資料，請確認代號是否正確。")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 統計指標
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

        # --- 關鍵修正：去除時區資訊 (Timezone-Naive) ---
        # 這樣才能跟 datetime.now() 做減法運算
        df.index = df.index.tz_localize(None) 
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]

    except Exception as e:
        st.error(f"數據處理錯誤: {e}")
        st.stop()

    # ==========================
    # 分頁 1: AI 多源分析
    # ==========================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta (波動係數)", f"{beta:.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='布林通道'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        st.markdown("---")
        col_news, col_ai = st.columns([1, 1])
        
        news_text_for_ai = ""
        ptt_text_for_ai = ""
        
        with col_news:
            st.subheader("📰 多源輿情偵測")
            
            st.markdown("**Google News 主流媒體**")
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                news_result = googlenews.result()[:4]
                if news_result:
                    for item in news_result:
                        st.write(f"- [{item['title']}]({item['link']})")
                        news_text_for_ai += f"{item['title']}\n"
                else:
                    st.caption("無近期主流新聞")
            except:
                st.caption("新聞連線失敗")
            
            st.markdown("**PTT 股版散戶熱議**")
            ptt_titles = fetch_ptt_sentiment(stock_name, limit=3)
            if ptt_titles:
                for t in ptt_titles:
                    st.write(f"- 💬 {t}")
                    ptt_text_for_ai += f"{t}\n"
            else:
                st.caption("無近期相關討論")

        with col_ai:
            st.subheader("🤖 Gemini 雙軌決策報告")
            with st.spinner("AI 正在進行思維鏈推論 (Chain of Thought)..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model_name, generation_config=genai.types.GenerationConfig(temperature=0.2))
                    
                    today_str = datetime.now().strftime("%Y年%m月%d日")
                    
                    suggested_weight = 50
                    if beta > 1.2:
                        suggested_weight = 70
                    elif beta < 0.8:
                        suggested_weight = 30
                    
                    prompt = f"""
                    你是一位專業量化交易員。今天是 {today_str}。
                    目標股票：{stock_name} ({ticker})，收盤價：{last_close}。
                    
                    ### 輸入數據
                    1. **技術指標**：RSI={df['RSI'].iloc[-1]:.2f}, MA20={df['MA20'].iloc[-1]:.2f}, Beta={beta:.2f}
                    2. **主流新聞**：\n{news_text_for_ai}
                    3. **社群論壇(PTT)**：\n{ptt_text_for_ai}
                    
                    ### 思考邏輯 (Chain of Thought)
                    1. 先分析 **Beta 值** 與 **社群熱度**，決定本股是「技術導向」還是「消息導向」。(建議消息權重基準：{suggested_weight}%)
                    2. 綜合主流媒體與散戶論壇的情緒，判斷市場共識。
                    3. 結合技術指標位置 (RSI高低檔)，推算合理目標價。
                    
                    請以 **純 JSON** 輸出，所有換行轉義為 \\n，內容繁體中文：
                    {{
                        "sentiment_weight": 70,
                        "weight_reason": "因為Beta高且PTT討論熱烈，故調高權重...",
                        "chart_data": {{ "target_price": 0, "high_price": 0, "low_price": 0, "buy_price": 0, "sell_price": 0 }},
                        "analysis_report": "## 分析報告... (Markdown格式)"
                    }}
                    """
                    response = model.generate_content(prompt)
                    clean_text = re.sub(r'```json|```', '', response.text).strip()
                    ai_data = json.loads(clean_text)
                    
                    if 'sentiment_weight' in ai_data:
                        w = ai_data['sentiment_weight']
                        st.info(f"⚖️ 消息權重: {w}% (Beta校正) | 技術權重: {100-w}%")
                        st.progress(w/100)
                        st.caption(f"判定理由：{ai_data.get('weight_reason', '無')}")
                    
                    if 'analysis_report' in ai_data:
                        st.markdown(ai_data['analysis_report'])
                        
                    if 'chart_data' in ai_data:
                        c = ai_data['chart_data']
                        now = datetime.now()
                        # 使用 naive date 計算，避免時區衝突
                        start_pt = now if (now - last_date).days > 1 else last_date
                        next_dt = now + timedelta(days=1)
                        while next_dt.weekday() > 4: next_dt += timedelta(days=1)
                        
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('high_price', last_close)], mode='lines+markers', line=dict(color='red', dash='dot'), name='樂觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('low_price', last_close)], mode='lines+markers', line=dict(color='green', dash='dot'), name='悲觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('target_price', last_close)], mode='lines+markers', line=dict(color='orange', width=4), name='目標'), row=1, col=1)

                except Exception as e:
                    st.error(f"分析錯誤: {e}")
        
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 分頁 2: 蒙地卡羅風險模擬
    # ==========================
    with tab2:
        st.header("🎲 蒙地卡羅風險模擬 (Monte Carlo Simulation)")
        st.caption("基於幾何布朗運動 (GBM) 模型，模擬未來股價路徑與風險值 (VaR)。")
        
        mc_col1, mc_col2 = st.columns([1, 3])
        
        try:
            log_returns, daily_volatility, drift, annual_volatility = calculate_metrics(df)
        except Exception as e:
            st.error(f"指標計算錯誤: {e}")
            st.stop()

        with mc_col1:
            st.subheader("參數設定")
            sim_days = st.slider("模擬天數", 30, 365, 90)
            n_simulations = st.slider("模擬次數", 100, 1000, 500)
            initial_investment = st.number_input("投資金額", value=100000, step=10000)
            
            st.markdown("---")
            st.metric("年化波動率", f"{annual_volatility*100:.2f}%")
            st.metric("日均漂移率 (Drift)", f"{drift*100:.4f}%")

        with mc_col2:
            if st.button("🎲 開始模擬運算"):
                with st.spinner("正在計算 1000+ 條平行宇宙路徑..."):
                    
                    last_price = last_close
                    all_paths = []
                    
                    for i in range(n_simulations):
                        daily_shocks = drift + daily_volatility * np.random.normal(0, 1, sim_days)
                        price_paths = [last_price]
                        for shock in daily_shocks:
                            price_paths.append(price_paths[-1] * np.exp(shock))
                        all_paths.append(price_paths)
                    
                    fig_mc = go.Figure()
                    x_axis = list(range(sim_days + 1))
                    
                    for path in all_paths[:100]:
                        fig_mc.add_trace(go.Scatter(
                            x=x_axis, y=path, 
                            mode='lines', 
                            line=dict(color='rgba(100, 100, 255, 0.05)', width=1), 
                            showlegend=False,
                            hovertemplate="第%{x}天: $%{y:.2f}"
                        ))
                    
                    avg_path = np.mean(all_paths, axis=0)
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', line=dict(color='orange', width=3), name='平均預期'))
                    
                    fig_mc.update_layout(title=f"未來 {sim_days} 天股價模擬", xaxis_title="天數", yaxis_title="股價", height=500)
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    final_prices = [p[-1] for p in all_paths]
                    expected_return = (np.mean(final_prices) - last_price) / last_price
                    
                    var_95_price = np.percentile(final_prices, 5)
                    loss_at_risk = (last_price - var_95_price) / last_price
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("預期報酬率", f"{expected_return*100:.2f}%")
                    r2.metric("95% VaR 風險值", f"-{loss_at_risk*100:.2f}%")
                    r3.metric("最差情況資產", f"${initial_investment * (1-loss_at_risk):,.0f}")
                    
                    st.markdown("### 🚦 風險監控儀表板")
                    if loss_at_risk > 0.15:
                        st.error(f"🚨 **高風險警報**：95% 機率虧損可能超過 15%！建議啟用熔斷機制或減少持倉。")
                    elif loss_at_risk > 0.08:
                        st.warning(f"⚠️ **中度風險**：波動較大，建議設置停損點。")
                    else:
                        st.success(f"✅ **低風險區域**：資產波動在安全範圍內。")
```

### 關鍵修正點：
我在程式碼第 158 行左右加了這句：
```python
df.index = df.index.tz_localize(None)
