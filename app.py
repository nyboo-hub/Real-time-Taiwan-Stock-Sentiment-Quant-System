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

# --- 1. 網頁基本設定 ---
# 設定網頁標題跟 layout，layout="wide" 這樣圖表比較不會被擠壓
st.set_page_config(page_title="AI 智能台股分析 v6.3", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統 (v6.3)")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)**、**蒙地卡羅模擬 (Risk)** 與 **Generative AI (多源輿情)** 的全方位決策系統。
> **技術架構**：Python ETL + Gemini LLM + Monte Carlo Simulation (Vectorized) + PTT Crawler
""")

# --- 側邊欄：系統參數 ---
st.sidebar.header("⚙️ 系統設定")

# 這裡做了一個 Demo 模式，方便 Demo 的時候如果網路不好可以直接秀
demo_mode = st.sidebar.toggle("🔥 啟用演示模式 (Demo Mode)", value=False, help="開啟後將使用模擬數據與預設 AI 回應，無需 API Key 即可展示功能。")

# --- 2. API Key 處理 ---
api_key = None

if demo_mode:
    # 如果是 Demo 模式，就隨便給個 key 讓程式能跑
    st.sidebar.success("✅ 目前處於演示模式")
    api_key = "demo_key" 
else:
    # 嘗試從 secrets 讀取 key，這樣部署的時候比較安全
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except:
        pass 

    # 如果沒讀到，就讓使用者自己輸入
    if not api_key:
        with st.sidebar.expander("🔐 API Key 設定", expanded=True):
            api_key = st.text_input("請輸入 Google Gemini API Key", type="password")
            st.caption("提示：部署到 Streamlit Cloud 後可設定 Secrets 隱藏此欄位")

# --- 3. 模型選擇 ---
selected_model_name = "gemini-1.5-flash" # 預設用 flash 比較快

if api_key and not demo_mode: 
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        
        # 這些是目前可以用的模型列表
        target_models = ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
        
        # 嘗試動態抓取 Google 目前開放的模型
        try:
            api_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            api_models = []
            
        # 把抓到的跟預設的合併並排序
        all_options = list(set(target_models + api_models))
        all_options.sort()
        
        # 我希望優先顯示比較新的模型，所以手動調整順序
        priorities = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for p in reversed(priorities):
            if p in all_options:
                all_options.remove(p) 
                all_options.insert(0, p) 

        selected_model_name = st.sidebar.selectbox("選擇推論模型 (Model)", all_options, index=0)
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

# --- 4. 股票輸入設定 ---
st.sidebar.header("📊 股票參數")

# 當股票代號改變時，嘗試自動抓取中文名稱
def update_stock_name():
    input_val = st.session_state.ticker_input.strip()
    code = input_val.split('.')[0]
    # 使用 twstock 套件來對照代號跟名稱
    if code in twstock.codes:
        st.session_state.stock_name_input = twstock.codes[code].name
    
    # 換股票的時候，要把之前的模擬結果清掉，不然圖會錯亂
    keys_to_clear = ['run_mc', 'mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

# 防呆機制：如果是純數字，幫使用者加上 .TW
if ticker.isdigit(): 
    ticker = f"{ticker}.TW"

# --- 5. 爬蟲與資料處理函數 ---

# 爬取 PTT Stock 版的標題
@st.cache_data(ttl=300) # 設定 cache 5分鐘，避免一直重複爬被鎖 IP
def fetch_ptt_sentiment(keyword, limit=5, retries=3):
    # 如果是 Demo 模式，直接回傳寫好的假資料
    if 'demo_mode' in globals() and demo_mode:
        return [f"[{keyword}] 營收創新高，散戶信心爆棚 (Demo)", f"[{keyword}] 外資調升目標價 (Demo)", f"[{keyword}] 技術面突破前高 (Demo)"]

    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'Cookie': 'over18=1'}
    
    # 加入 retry 機制，網路不穩的時候多試幾次
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
            # 失敗的話休息一下再試
            if attempt < retries - 1:
                time.sleep(1)
                continue
    return [] # 真的爬不到就回傳空串列

# 產生模擬資料 (當 Yahoo Finance 掛掉的時候用)
def generate_mock_data(days=120):
    # 用常態分佈隨機產生股價，讓圖表看起來像真的
    dates = pd.date_range(end=datetime.now(), periods=days).normalize()
    price = 1000
    prices = []
    for _ in range(days):
        change = np.random.normal(0, 15)
        price += change
        if price < 100: price = 100 # 防止跌到變負數
        prices.append(price)
    
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    # 簡單模擬一下 OHLC，讓 K 線圖畫得出來
    df['Open'] = [p + np.random.normal(0, 5) for p in prices]
    df['High'] = [max(o, c) + abs(np.random.normal(0, 10)) for o, c in zip(df['Open'], df['Close'])]
    df['Low'] = [min(o, c) - abs(np.random.normal(0, 10)) for o, c in zip(df['Open'], df['Close'])]
    return df

# 計算各種統計指標
@st.cache_data
def calculate_metrics(df):
    close = df['Close'].ffill()
    # 計算對數報酬率 log returns
    log_returns = np.log(close / close.shift(1))
    
    u = log_returns.mean() 
    var = log_returns.var() 
    daily_volatility = log_returns.std() 
    
    # 計算漂移率 Drift
    drift = u - (0.5 * var)
    # 年化波動率 (假設一年交易日 252 天)
    annual_volatility = daily_volatility * np.sqrt(252)
    
    return log_returns, daily_volatility, drift, annual_volatility

# --- 蒙地卡羅模擬 (這裡原本用 for loop 跑很慢，改用 NumPy 加速) ---
def run_vectorized_monte_carlo(last_price, drift, daily_vol, sim_days, n_sims):
    # 1. 一次生成所有路徑需要的隨機變數 (矩陣大小: 模擬次數 x 天數)
    random_shocks = np.random.normal(0, 1, (n_sims, sim_days))
    
    # 2. 透過矩陣運算一次算出每天的漲跌倍數
    daily_returns = np.exp(drift + daily_vol * random_shocks)
    
    # 3. 初始化價格路徑矩陣
    price_paths = np.zeros((n_sims, sim_days + 1))
    price_paths[:, 0] = last_price # 起始點都是最後收盤價
    
    # 4. 用累積乘積 (cumprod) 算出每天的價格
    # 這裡比跑迴圈快非常多 (Vectorization)
    price_paths[:, 1:] = last_price * np.cumprod(daily_returns, axis=1)
    
    return price_paths

# --- 6. 主程式開始 ---

# 初始化 session state 變數
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False
if 'run_mc' not in st.session_state:
    st.session_state.run_mc = False

def start_analysis_callback():
    st.session_state['analysis_started'] = True

# 1. 啟動按鈕
st.button("🚀 啟動全方位分析", on_click=start_analysis_callback)

# 2. 建立分頁 Tabs
tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬 (Risk Lab)"])

# --- Tab 2: 說明文字 ---
with tab2:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #ff9966, #ff5e62); color: white; padding: 20px; border-radius: 15px; margin: 0px 0px 20px 0px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="margin-top:0; color: white; text-shadow: 1px 1px 2px black;">🎲 蒙地卡羅白話解釋</h3>
            <div style="text-align: left; display: inline-block; background: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px;">
                <b>為什麼圖長這樣？</b><br>
                1. <b>淡藍色線</b>：我模擬了 1000 個平行宇宙的可能走勢。<br>
                2. <b>橘色粗線</b>：平均下來的預期趨勢。<br>
                3. <b>95% VaR</b>：統計學上的「風險值」，代表最慘的情況可能會虧多少。<br>
            </div>
            <br>
            <b style="font-size: 1.2em;">使用 NumPy 向量化運算，模擬速度提升 100 倍！</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not st.session_state['analysis_started']:
        st.warning("👈 請先點擊上方「🚀 啟動全方位分析」按鈕")

if not st.session_state['analysis_started']:
    with tab1:
        st.info("👈 請在左側設定參數，並點擊上方「🚀 啟動全方位分析」按鈕開始。")

# 4. 開始執行分析邏輯
if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。")
        st.stop() 

    # --- 資料處理 (ETL) ---
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 判斷是否為 Demo 模式
        if demo_mode:
             df = generate_mock_data(days)
             beta = 1.3 
             st.toast("🔥 目前處於演示模式 (Using Mock Data)", icon="🧪")
        else:
            try:
                # 嘗試抓取真實資料
                stock_obj = yf.Ticker(ticker)
                df = stock_obj.history(start=start_date, end=end_date)
                
                if df.empty or len(df) < 5:
                    raise ValueError("Data empty")
                
                # 嘗試抓取 Beta 值
                try:
                    stock_info = stock_obj.info
                    beta = stock_info.get('beta', 1.0) if stock_info else 1.0
                except:
                    beta = 1.0 # 抓不到就用預設值

            except Exception as e:
                # 如果連線失敗，自動切換到 Demo 模式，防止程式崩潰
                st.toast(f"⚠️ 連線失敗，自動切換至演示模式", icon="🛡️")
                df = generate_mock_data(days)
                beta = 1.2
            
        # 處理 MultiIndex 的問題 (yfinance 有時候會回傳這種格式)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 計算技術指標
        df['MA5'] = df['Close'].rolling(window=5).mean()   
        df['MA20'] = df['Close'].rolling(window=20).mean() 
        df['STD'] = df['Close'].rolling(window=20).std()   
        df['Upper'] = df['MA20'] + (2 * df['STD']) # 布林通道上緣
        df['Lower'] = df['MA20'] - (2 * df['STD']) # 布林通道下緣
        
        # 計算 RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 去除時區資訊，避免繪圖錯誤
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]

    except Exception as e:
        st.error(f"數據處理錯誤: {e}")
        st.stop()

    # ==========================
    # 分頁 1: AI 多源分析介面
    # ==========================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta", f"{beta:.2f}")

        # 繪製 K 線圖 + 技術指標
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='布林通道'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col_news, col_ai = st.columns([1, 1])
        
        news_text_for_ai = ""
        ptt_text_for_ai = ""
        
        with col_news:
            st.subheader("📰 多源輿情偵測")
            
            if demo_mode:
                # 演示用的假新聞
                st.markdown("**[Demo] 主流媒體 & PTT**")
                st.write(f"- 📰 {stock_name} 營收創新高 (Demo)")
                st.write(f"- 💬 {stock_name} 歐印了啦 (Demo)")
                news_text_for_ai = "營收創新高，外資喊買。"
                ptt_text_for_ai = "散戶信心爆棚，歐印。"
            else:
                # 串接 Google News API
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
                
                # 顯示 PTT 爬蟲結果
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
                
                if demo_mode:
                    time.sleep(1) # 假裝在思考
                    ai_data = {
                        "sentiment_weight": 75,
                        "weight_reason": "[Demo] Beta偏高且熱度高",
                        "analysis_report": f"## {stock_name} 分析報告 (Demo)\n\n建議：偏多操作。\n*此為演示數據*",
                        "chart_data": {"target_price": last_close*1.05, "high_price": last_close*1.08, "low_price": last_close*0.95}
                    }
                else:
                    try:
                        # 呼叫 Gemini 模型
                        model = genai.GenerativeModel(selected_model_name, generation_config=genai.types.GenerationConfig(temperature=0.2))
                        today_str = datetime.now().strftime("%Y年%m月%d日")
                        
                        # 根據 Beta 值調整消息面的權重建議
                        suggested_weight = 50
                        if beta > 1.2: suggested_weight = 70
                        elif beta < 0.8: suggested_weight = 30
                        
                        # 這是給 AI 的 Prompt
                        prompt = f"""
                        你是一位專業量化交易員。今天是 {today_str}。
                        目標股票：{stock_name} ({ticker})，收盤價：{last_close:.2f}。
                        
                        ### 輸入數據
                        1. **技術指標**：RSI={df['RSI'].iloc[-1]:.2f}, MA20={df['MA20'].iloc[-1]:.2f}, Beta={beta:.2f}
                        2. **主流新聞**：\n{news_text_for_ai}
                        3. **社群論壇(PTT)**：\n{ptt_text_for_ai}
                        
                        請以 **純 JSON** 輸出，確保格式正確：
                        {{
                            "sentiment_weight": {suggested_weight},
                            "weight_reason": "理由...",
                            "chart_data": {{ "target_price": 0, "high_price": 0, "low_price": 0, "buy_price": 0, "sell_price": 0 }},
                            "analysis_report": "## Markdown 報告內容..."
                        }}
                        """
                        response = model.generate_content(prompt)
                        # 清理回傳的文字，確保是 JSON 格式
                        clean_text = re.sub(r'```json|```', '', response.text).strip()
                        ai_data = json.loads(clean_text)
                        
                    except Exception as e:
                        st.error(f"AI 分析失敗: {e}")
                        ai_data = {"analysis_report": "AI 連線失敗，請檢查 API Key。", "chart_data": {}}

                # 顯示 AI 分析結果
                if 'sentiment_weight' in ai_data:
                    w = ai_data['sentiment_weight']
                    st.info(f"⚖️ 消息權重: {w}% (Beta校正) | 技術權重: {100-w}%")
                    st.progress(w/100)
                
                if 'analysis_report' in ai_data:
                    st.markdown(ai_data['analysis_report'])
                
                # 在圖表上畫出 AI 預測點位
                if 'chart_data' in ai_data:
                    c = ai_data['chart_data']
                    now = datetime.now()
                    start_pt = now if (now - last_date).days > 1 else last_date
                    next_dt = now + timedelta(days=1)
                    while next_dt.weekday() > 4: next_dt += timedelta(days=1)
                    
                    fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('high_price', last_close)], mode='lines+markers', line=dict(color='red', dash='dot'), name='樂觀'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('low_price', last_close)], mode='lines+markers', line=dict(color='green', dash='dot'), name='悲觀'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('target_price', last_close)], mode='lines+markers', line=dict(color='orange', width=4), name='目標'), row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 分頁 2: 蒙地卡羅風險模擬 (修正後)
    # ==========================
    with tab2:
        st.divider() 
        mc_col1, mc_col2 = st.columns([1, 3])
        
        try:
            log_returns, daily_volatility, drift, annual_volatility = calculate_metrics(df)
        except Exception as e:
            # 如果數據不足，先用預設參數跑，避免當機
            annual_volatility = 0.3
            drift = 0.0005
            daily_volatility = 0.02
            st.warning("⚠️ 使用預設波動率參數 (Demo Mode)")

        with mc_col1:
            with st.form("mc_form"):
                st.subheader("參數設定")
                sim_days = st.slider("模擬天數", 30, 365, 90)
                # 現在用向量化運算，模擬 5000 次也很快
                n_simulations = st.slider("模擬次數", 100, 5000, 1000)
                initial_investment = st.number_input("投資金額", value=100000, step=10000)
                st.metric("年化波動率", f"{annual_volatility*100:.2f}%")
                st.metric("日均漂移率 (Drift)", f"{drift*100:.4f}%")
                submitted = st.form_submit_button("🎲 開始模擬運算 (Vectorized)", type="primary", use_container_width=True)

        with mc_col2:
            if submitted:
                with st.spinner(f"正在平行運算 {n_simulations} 條市場路徑..."):
                    
                    # 使用前面寫好的 NumPy 加速函數
                    all_paths = run_vectorized_monte_carlo(last_close, drift, daily_volatility, sim_days, n_simulations)
                    
                    fig_mc = go.Figure()
                    x_axis = list(range(sim_days + 1))
                    
                    # 只要畫前 100 條路徑就好，不然瀏覽器會跑不動
                    subset_paths = all_paths[:100]
                    for path in subset_paths:
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=path, mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)', width=1), showlegend=False))
                    
                    # 畫出平均預期線
                    avg_path = np.mean(all_paths, axis=0)
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', line=dict(color='orange', width=3), name='平均預期'))
                    
                    fig_mc.update_layout(title=f"未來 {sim_days} 天股價模擬 ({n_simulations} 次運算)", xaxis_title="天數", yaxis_title="股價", height=500)
                    
                    # 計算 VaR 風險值 (取第 5 百分位數)
                    final_prices = all_paths[:, -1]
                    loss_at_risk = (last_close - np.percentile(final_prices, 5)) / last_close
                    expected_return = (np.mean(final_prices) - last_close) / last_close
                    
                    # 把結果存到 session_state，這樣切換 tab 才不會消失
                    st.session_state.mc_fig = fig_mc
                    st.session_state.mc_return = expected_return
                    st.session_state.mc_risk = loss_at_risk
                    st.session_state.mc_asset = initial_investment * (1 - loss_at_risk)
                    st.session_state.run_mc = True

            # 如果已經跑過模擬，就顯示結果
            if st.session_state.run_mc and 'mc_fig' in st.session_state:
                st.plotly_chart(st.session_state.mc_fig, use_container_width=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("預期報酬率", f"{st.session_state.mc_return*100:.2f}%")
                r2.metric("95% VaR 風險值", f"-{st.session_state.mc_risk*100:.2f}%")
                r3.metric("最差情況資產", f"${st.session_state.mc_asset:,.0f}")
                
                # 風險警示燈號
                risk = st.session_state.mc_risk
                if risk > 0.15:
                    st.error("🚨 高風險警報：虧損可能超過 15%！")
                elif risk > 0.08:
                    st.warning("⚠️ 中度風險：建議設停損。")
                else:
                    st.success("✅ 低風險區域。")
                st.divider()
                # 放在 mc_col2 裡面比較整齊
                if st.button("清除模擬結果", type="secondary", use_container_width=True):
                    st.session_state.run_mc = False
                    keys_to_clean = ['mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
                    for k in keys_to_clean:
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
