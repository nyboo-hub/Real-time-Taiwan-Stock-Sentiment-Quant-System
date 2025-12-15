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
st.set_page_config(page_title="AI 智能台股分析 v6.0 (Demo Ready)", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統 (v6.0)")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)**、**蒙地卡羅模擬 (Risk)** 與 **Generative AI (多源輿情)** 的全方位決策系統。
> **技術架構**：Python ETL + Gemini LLM + Monte Carlo Simulation + PTT Crawler
""")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 系統設定")

# 🔥 新增：演示模式開關 (一鍵切換)
demo_mode = st.sidebar.toggle("🔥 啟用演示模式 (Demo Mode)", value=False, help="開啟後將使用模擬數據與預設 AI 回應，無需 API Key 即可展示功能。")

if demo_mode:
    st.sidebar.success("✅ 目前處於演示模式")
    api_key = "demo_key" # 給個假 Key 讓流程繼續
else:
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

if api_key and not demo_mode: 
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
    
    keys_to_clear = ['run_mc', 'mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

if ticker.isdigit(): 
    ticker = f"{ticker}.TW"

# --- 5. 核心功能函數 ---

@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, limit=5, retries=3):
    if demo_mode:
        return [f"[{keyword}] 營收創新高，散戶信心爆棚 (Demo)", f"[{keyword}] 外資調升目標價 (Demo)", f"[{keyword}] 技術面突破前高 (Demo)"]

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

# 產生模擬股價資料
def generate_mock_data(days=120):
    dates = pd.date_range(end=datetime.now(), periods=days).normalize()
    price = 1000
    prices = []
    for _ in range(days):
        change = np.random.normal(0, 15)
        price += change
        if price < 100: price = 100
        prices.append(price)
    
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = [p + np.random.normal(0, 5) for p in prices]
    df['High'] = [max(o, c) + abs(np.random.normal(0, 10)) for o, c in zip(df['Open'], df['Close'])]
    df['Low'] = [min(o, c) - abs(np.random.normal(0, 10)) for o, c in zip(df['Open'], df['Close'])]
    return df

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

# --- 6. 主程式邏輯 (Main Loop) ---

if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False
if 'run_mc' not in st.session_state:
    st.session_state.run_mc = False

def start_analysis_callback():
    st.session_state['analysis_started'] = True

# 1. 建立按鈕
st.button("🚀 啟動全方位分析", on_click=start_analysis_callback)

# 2. 建立分頁
tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬 (Risk Lab)"])

# --- Tab 2: 蒙地卡羅說明 (炫彩版回歸！) ---
with tab2:
    # ────────── 這是你指定的炫彩漸層說明框 ──────────
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #ff9966, #ff5e62); color: white; padding: 20px; border-radius: 15px; margin: 0px 0px 20px 0px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="margin-top:0; color: white; text-shadow: 1px 1px 2px black;">🎲 蒙地卡羅白話解釋</h3>
            <div style="text-align: left; display: inline-block; background: rgba(0,0,0,0.1); padding: 15px; border-radius: 10px;">
                <b>為什麼圖長這樣？</b><br>
                1. <b>淡藍色線像扇子越張越開</b> → 時間越久未來越不確定<br>
                2. <b>橘色粗線 = 500 次平均</b> → 這檔股票真正的長期趨勢<br>
                3. <b>95% VaR = 最慘 5% 的情況</b> → 銀行、金控每天都在看這個數字<br>
                4. <b>紅黃綠燈 = 實務風控標準</b> → 紅燈表示老闆會叫你停損！<br>
            </div>
            <br><br>
            <b style="font-size: 1.2em; background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 5px;">簡單說：這是用 500 個平行宇宙幫你預演未來會不會爆倉！</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ───────────────────────────────────────
    
    st.caption("基於幾何布朗運動 (GBM) 模型，符合國際量化交易標準")

    if not st.session_state['analysis_started']:
        st.warning("👈 請先點擊上方「🚀 啟動全方位分析」按鈕，載入股票資料後即可開始模擬～")

# --- Tab 1: 尚未開始時的提示 ---
if not st.session_state['analysis_started']:
    with tab1:
        st.info("👈 請在左側設定參數，並點擊上方「🚀 啟動全方位分析」按鈕開始。")

# 4. 執行分析
if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。")
        st.stop() 

    # --- 共用資料處理 (ETL) ---
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 邏輯：如果是 Demo 模式，直接用假資料
        if demo_mode:
             df = generate_mock_data(days)
             beta = 1.3 # Demo 預設一個較高的 Beta 讓權重比較好看
             st.toast("🔥 目前處於演示模式 (Using Mock Data)", icon="🧪")
        else:
            try:
                stock_obj = yf.Ticker(ticker)
                df = stock_obj.history(start=start_date, end=end_date)
                if df.empty or len(df) < 5:
                    raise ValueError("Data empty")
                
                try:
                    stock_info = stock_obj.info
                    beta = stock_info.get('beta', 1.0) if stock_info else 1.0
                    if beta is None: beta = 1.0
                except:
                    beta = 1.0

            except Exception as e:
                # 自動 fallback 到 Demo 模式
                st.toast(f"⚠️ 連線失敗，自動切換至演示模式", icon="🛡️")
                df = generate_mock_data(days)
                beta = 1.2
            
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
        st.error(f"嚴重系統錯誤: {e}")
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
            
            # 如果是演示模式，顯示假新聞
            if demo_mode:
                st.markdown("**Google News 主流媒體**")
                st.caption("⚠️ 演示模式：模擬新聞數據")
                st.write(f"- [{stock_name} 營收創新高，外資喊買 (Demo)](https://google.com)")
                st.write(f"- [{stock_name} 法說會報喜，股價強勢 (Demo)](https://google.com)")
                news_text_for_ai = f"{stock_name} 營收創新高，外資喊買。\n{stock_name} 法說會報喜，股價強勢。"
                
                st.markdown("**PTT 股版散戶熱議**")
                st.write(f"- 💬 {stock_name} 這波穩了嗎？ (Demo)")
                st.write(f"- 💬 {stock_name} 歐印了啦 (Demo)")
                ptt_text_for_ai = f"{stock_name} 這波穩了嗎？\n{stock_name} 歐印了啦"
            else:
                # 正常模式
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
                    # 如果是演示模式，回傳假 JSON
                    if demo_mode:
                        time.sleep(2) # 假裝思考
                        ai_data = {
                            "sentiment_weight": 75,
                            "weight_reason": "【演示模式】偵測到 Beta 值高 (1.3) 且社群討論熱度高，判定為消息面主導。",
                            "chart_data": { 
                                "target_price": last_close * 1.03, 
                                "high_price": last_close * 1.05, 
                                "low_price": last_close * 0.98, 
                                "buy_price": last_close * 0.99, 
                                "sell_price": last_close * 1.04 
                            },
                            "analysis_report": f"## {stock_name} 雙軌分析報告 (演示版)\n\n1. **技術面分析**：股價站上均線，RSI 指標 ({df['RSI'].iloc[-1]:.2f}) 顯示動能強勁。\n2. **市場情緒**：主流媒體與 PTT 皆呈現看多趨勢。\n3. **預測**：短期內有望挑戰前高。\n\n*註：此為演示模式生成之模擬數據。*"
                        }
                    else:
                        # 正常呼叫 API
                        model = genai.GenerativeModel(selected_model_name, generation_config=genai.types.GenerationConfig(temperature=0.2))
                        
                        today_str = datetime.now().strftime("%Y年%m月%d日")
                        suggested_weight = 50
                        if beta > 1.2: suggested_weight = 70
                        elif beta < 0.8: suggested_weight = 30
                        
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
                        start_pt = now if (now - last_date).days > 1 else last_date
                        next_dt = now + timedelta(days=1)
                        while next_dt.weekday() > 4: next_dt += timedelta(days=1)
                        
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('high_price', last_close)], mode='lines+markers', line=dict(color='red', dash='dot'), name='樂觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('low_price', last_close)], mode='lines+markers', line=dict(color='green', dash='dot'), name='悲觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('target_price', last_close)], mode='lines+markers', line=dict(color='orange', width=4), name='目標'), row=1, col=1)

                except Exception as e:
                    st.error(f"AI 分析失敗: {e}")
                    st.markdown(f"**系統提示**：AI 連線不穩定，但根據技術指標 RSI={df['RSI'].iloc[-1]:.2f}，建議區間操作。")
        
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 分頁 2: 蒙地卡羅風險模擬
    # ==========================
    with tab2:
        st.divider() 
        mc_col1, mc_col2 = st.columns([1, 3])
        try:
            log_returns, daily_volatility, drift, annual_volatility = calculate_metrics(df)
        except Exception as e:
            # Mock Metrics
            annual_volatility = 0.3
            drift = 0.0005
            st.warning("⚠️ 使用預設波動率參數 (Demo Mode)")

        with mc_col1:
            st.subheader("參數設定")
            sim_days = st.slider("模擬天數", 30, 365, 90)
            n_simulations = st.slider("模擬次數", 100, 1000, 500)
            initial_investment = st.number_input("投資金額", value=100000, step=10000)
            st.metric("年化波動率", f"{annual_volatility*100:.2f}%")
            st.metric("日均漂移率 (Drift)", f"{drift*100:.4f}%")

        with mc_col2:
            if st.button("🎲 開始模擬運算", type="primary", use_container_width=True):
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
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=path, mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)', width=1), showlegend=False, hovertemplate="第%{x}天: $%{y:.2f}"))
                    
                    avg_path = np.mean(all_paths, axis=0)
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', line=dict(color='orange', width=3), name='平均預期'))
                    
                    fig_mc.update_layout(title=f"未來 {sim_days} 天股價模擬 ({n_simulations} 次運算)", xaxis_title="天數", yaxis_title="股價", height=500)
                    
                    final_prices = [p[-1] for p in all_paths]
                    expected_return = (np.mean(final_prices) - last_price) / last_price
                    var_95_price = np.percentile(final_prices, 5)
                    loss_at_risk = (last_price - var_95_price) / last_price
                    
                    st.session_state.mc_fig = fig_mc
                    st.session_state.mc_return = expected_return
                    st.session_state.mc_risk = loss_at_risk
                    st.session_state.mc_asset = initial_investment * (1-loss_at_risk)
                    st.session_state.run_mc = True

            if st.session_state.run_mc and 'mc_fig' in st.session_state:
                st.plotly_chart(st.session_state.mc_fig, use_container_width=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("預期報酬率", f"{st.session_state.mc_return*100:.2f}%")
                r2.metric("95% VaR 風險值", f"-{st.session_state.mc_risk*100:.2f}%")
                r3.metric("最差情況資產", f"${st.session_state.mc_asset:,.0f}")
                
                risk = st.session_state.mc_risk
                if risk > 0.15:
                    st.error("🚨 高風險警報：虧損可能超過 15%！")
                elif risk > 0.08:
                    st.warning("⚠️ 中度風險：建議設停損。")
                else:
                    st.success("✅ 低風險區域。")
            
            with col_clear:
                if st.session_state.run_mc:
                    if st.button("清除模擬結果"):
                        st.session_state.run_mc = False
                        st.rerun()
