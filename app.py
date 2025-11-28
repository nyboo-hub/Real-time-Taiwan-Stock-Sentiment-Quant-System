# ==========================================
# 區塊 1: 匯入工具箱 (Import Libraries)
# ==========================================
import streamlit as st          # 這是做網頁介面的神器，就像用積木蓋房子
import yfinance as yf           # 這是負責去 Yahoo Finance 抓股票資料的快遞員
import pandas as pd             # 這是 Python 的 Excel，專門處理表格數據
import numpy as np              # 這是數學計算機，處理矩陣、標準差、log運算
import plotly.graph_objects as go # 這是畫圖工具，畫出漂亮的互動式 K 線圖
from plotly.subplots import make_subplots # 這是用來把兩張圖 (K線 + RSI) 拼在一起的工具
from GoogleNews import GoogleNews # 這是去 Google 新聞抓標題的爬蟲
import google.generativeai as genai # 這是 Google Gemini AI 的大腦
from datetime import datetime, timedelta # 這是處理時間的工具 (今天幾號、昨天幾號)
import json     # 這是處理資料格式的工具 (AI 回傳的資料通常是 JSON)
import re       # 這是「正規表達式」，用來在亂七八糟的文字裡抓出我們要把的重點
import twstock  # 這是台灣股市的工具，用來查股票代號對應的名稱
import requests # 這是發送網路請求的工具 (爬蟲用)
from bs4 import BeautifulSoup # 這是把爬下來的網頁原始碼 (HTML) 整理乾淨的工具
import time     # 這是控制時間的 (例如暫停幾秒、重試)

# ==========================================
# 區塊 2: 網頁基礎設定
# ==========================================
# 設定網頁標題、圖標，layout="wide" 代表使用寬螢幕模式
st.set_page_config(page_title="AI 智能台股情緒量化分析系統", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)**、**蒙地卡羅模擬 (Risk)** 與 **Generative AI (多源輿情)** 的全方位決策系統。
> **技術架構**：Python ETL + Gemini LLM + Monte Carlo Simulation + PTT Crawler
""")

# ==========================================
# 區塊 3: API 金鑰管理 (資安防護)
# ==========================================
api_key = None
# try-except 是「錯誤處理」。意思是：試試看做這件事，如果報錯了不要當機，執行 except 裡面的事
try:
    # 嘗試從 Streamlit Cloud 的秘密庫 (Secrets) 拿密碼
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except:
    pass # 如果本機沒有 secrets 檔案，就跳過，什麼都不做 (pass)

# 如果沒抓到金鑰 (代表在本機執行)，就顯示輸入框讓使用者自己貼
if not api_key:
    with st.sidebar.expander("🔐 API Key 設定", expanded=True):
        api_key = st.text_input("請輸入 Google Gemini API Key", type="password")
        st.caption("提示：部署到 Streamlit Cloud 後可設定 Secrets 隱藏此欄位")

# ==========================================
# 區塊 4: AI 模型選擇器 (下拉選單)
# ==========================================
# 預設選用這隻開源模型，這是你的策略
selected_model_name = "gemma-3n-e4b-it"

if api_key: # 只有當使用者填了 Key 之後，才顯示模型設定
    st.sidebar.header("🤖 AI 模型設定")
    try:
        # 設定 AI 的金鑰，讓 Google 知道你是誰
        genai.configure(api_key=api_key)
        
        # 定義我們想要用的模型清單
        target_models = [
            'gemma-3n-e4b-it',              
            'gemini-2.5-pro-preview-03-25', 
            'gemini-1.5-pro',               
            'gemini-1.5-flash',             
            'gemini-pro'                    
        ]
        
        # 嘗試去問 Google 目前有哪些模型可用 (List Models)
        try:
            # 這一行有點複雜：它用「列表推導式」把 Google 回傳的亂碼整理成乾淨的名字
            api_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            api_models = []
            
        # 把我們想要的跟 Google 提供的合併，並用 set() 去除重複
        all_options = list(set(target_models + api_models))
        all_options.sort()
        
        # 手動調整排序，把你最想秀的模型排在最上面
        priorities = ['gemma-3n-e4b-it', 'gemini-2.5-pro-preview-03-25', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for p in reversed(priorities):
            if p in all_options:
                all_options.remove(p) # 先移除
                all_options.insert(0, p) # 再插到第 0 位 (最前面)

        # 顯示下拉選單
        selected_model_name = st.sidebar.selectbox("選擇推論模型 (Model)", all_options, index=0)
        
        # 根據選到的模型，顯示不同的提示訊息 (UX 優化)
        if "gemma" in selected_model_name:
            st.sidebar.warning(f"🧪 已啟用實驗性模型: {selected_model_name}")
        elif "preview" in selected_model_name:
            st.sidebar.success(f"🚀 已啟用最新預覽版: {selected_model_name}")
        elif "flash" in selected_model_name:
            st.sidebar.info(f"⚡ 已啟用高速推論模式")
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

# ==========================================
# 區塊 5: 股票參數輸入 (前端互動)
# ==========================================
st.sidebar.header("📊 股票參數")

# 這是一個函式，用來實現「輸入代號 -> 自動跳出名稱」的功能
def update_stock_name():
    # 取得使用者輸入的內容，去頭去尾 (strip)
    input_val = st.session_state.ticker_input.strip()
    # 取得小數點前面的數字 (例如 2330.TW -> 2330)
    code = input_val.split('.')[0]
    # 如果這個代號在 twstock 的資料庫裡
    if code in twstock.codes:
        # 就把名稱填入 stock_name_input 變數裡
        st.session_state.stock_name_input = twstock.codes[code].name
    
    # 切換股票時，順便把之前的蒙地卡羅運算結果清掉，以免圖表混淆
    keys_to_clear = ['run_mc', 'mc_fig', 'mc_return', 'mc_risk', 'mc_asset']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# 顯示輸入框
# key="ticker_input" 是給上面那個函式用的 ID
# on_change=update_stock_name 代表「當內容改變時，執行 update_stock_name 函式」
ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

# 防呆機制：如果使用者忘記打 .TW，幫他補上
if ticker.isdigit(): # 如果全是數字
    ticker = f"{ticker}.TW"

# ==========================================
# 區塊 6: 核心功能函數定義 (後端邏輯)
# ==========================================

# @st.cache_data 是一個「裝飾器」。它的作用是「快取 (Cache)」。
# 如果輸入的 keyword 一樣，它就不會真的去爬蟲，而是直接回傳上次的結果。
# ttl=300 代表快取存活 300 秒 (5分鐘)，避免每次按按鈕都要等。
@st.cache_data(ttl=300)
def fetch_ptt_sentiment(keyword, limit=5, retries=3):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    # 偽裝成瀏覽器 (User-Agent)，不然 PTT 會擋爬蟲
    headers = {'User-Agent': 'Mozilla/5.0 ...', 'Cookie': 'over18=1'} # over18=1 是為了過 PTT 的滿18歲檢查
    
    # 重試機制 (Retry Logic)
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200: # 200 代表成功
                soup = BeautifulSoup(response.text, 'html.parser') # 解析 HTML
                titles = soup.find_all('div', class_='title') # 找到所有標題區塊
                result = []
                for t in titles[:limit]:
                    a_tag = t.find('a') # 找到標題連結
                    if a_tag:
                        result.append(a_tag.text.strip()) # 抓出文字
                return result
        except Exception:
            # 如果失敗，等待 1 秒再試 (Backoff)
            if attempt < retries - 1:
                time.sleep(1)
                continue
    return []

# 計算波動率的函式 (給蒙地卡羅用的)
@st.cache_data
def calculate_metrics(df):
    # ffill() 是 "Forward Fill"，如果今天資料缺漏，就用昨天的填補
    close = df['Close'].ffill()
    # 計算「對數報酬率 (Log Returns)」
    # 這是金融工程的標準做法，因為股價是連續複利
    log_returns = np.log(close / close.shift(1))
    
    u = log_returns.mean() # 平均報酬
    var = log_returns.var() # 變異數
    daily_volatility = log_returns.std() # 日波動率 (標準差)
    
    # 計算漂移項 (Drift)：這是股價長期趨勢的動能
    # 公式：Drift = 平均報酬 - (變異數的一半)
    drift = u - (0.5 * var)
    # 年化波動率 (乘以根號 252 天)
    annual_volatility = daily_volatility * np.sqrt(252)
    
    return log_returns, daily_volatility, drift, annual_volatility

# ==========================================
# 區塊 7: 主程式邏輯 (Main Loop)
# ==========================================

# 初始化 session_state (網頁暫存記憶體)
# 這樣就算網頁重整，程式也知道「分析按鈕」是不是曾經被按過
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False

def start_analysis_callback():
    st.session_state['analysis_started'] = True

# 顯示這顆大按鈕
st.button("🚀 啟動全方位分析", on_click=start_analysis_callback)

# 如果按鈕被按過，才執行下面的東西
if st.session_state['analysis_started']:
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。")
        st.stop() # 停止執行

    # 建立兩個分頁
    tab1, tab2 = st.tabs(["🤖 AI 多源輿情決策", "🎲 蒙地卡羅風險模擬 (Risk Lab)"])

    # --- 共用資料處理 (ETL) ---
    try:
        # 設定日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 使用 yfinance 下載資料
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date, end=end_date)
        
        # 抓 Beta 值 (如果抓不到就預設 1.0)
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
            st.error(f"找不到 {ticker} 的股價資料。")
            st.stop()
            
        # 處理 MultiIndex (這是 yfinance 新版的格式問題，要修正它)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # === 統計指標計算 (Pandas 應用) ===
        df['MA5'] = df['Close'].rolling(window=5).mean()   # 週線
        df['MA20'] = df['Close'].rolling(window=20).mean() # 月線
        df['STD'] = df['Close'].rolling(window=20).std()   # 標準差
        df['Upper'] = df['MA20'] + (2 * df['STD']) # 布林通道上軌
        df['Lower'] = df['MA20'] - (2 * df['STD']) # 布林通道下軌
        
        # RSI 計算 (比較複雜的公式)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # === 時間格式修正 (Bug Fix) ===
        # 因為 yfinance 的時間有時區 (UTC)，但 datetime.now() 沒有
        # 兩個不一樣格式的時間不能相減，所以要把時區拿掉 (tz_localize(None))
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]

    except Exception as e:
        st.error(f"數據處理錯誤: {e}")
        st.stop()

    # ==========================
    # 分頁 1: AI 多源分析 (內容顯示)
    # ==========================
    with tab1:
        # 顯示四個大數字
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}")
        c2.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
        c3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        c4.metric("Beta (波動係數)", f"{beta:.2f}")

        # 畫 K 線圖 (使用 Plotly)
        # make_subplots 是為了讓 K 線圖在上面，RSI 在下面
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
        # ... (略過重複的繪圖代碼) ...
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        # 畫 RSI 的 70/30 分界線
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        st.markdown("---")
        col_news, col_ai = st.columns([1, 1])
        
        news_text_for_ai = ""
        ptt_text_for_ai = ""
        
        with col_news:
            st.subheader("📰 多源輿情偵測")
            # 呼叫 Google News
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                news_result = googlenews.result()[:4]
                if news_result:
                    for item in news_result:
                        st.write(f"- [{item['title']}]({item['link']})")
                        news_text_for_ai += f"{item['title']}\n"
            except:
                st.caption("新聞連線失敗")
            
            # 呼叫我們剛剛寫的 PTT 爬蟲
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
                    # 設定 AI 參數：temperature=0.2 代表要它「冷靜、客觀」，不要亂編故事
                    model = genai.GenerativeModel(selected_model_name, generation_config=genai.types.GenerationConfig(temperature=0.2))
                    
                    # 組合 Prompt (提示詞工程)
                    prompt = f"""
                    你是一位專業量化交易員。
                    目標股票：{stock_name} ({ticker})，收盤價：{last_close}。
                    
                    ### 輸入數據
                    1. **技術指標**：RSI={df['RSI'].iloc[-1]:.2f}, MA20={df['MA20'].iloc[-1]:.2f}, Beta={beta:.2f}
                    2. **主流新聞**：\n{news_text_for_ai}
                    3. **社群論壇(PTT)**：\n{ptt_text_for_ai}
                    
                    ### 思考邏輯 (Chain of Thought)
                    1. 先分析 **Beta 值**，決定本股是「技術導向」還是「消息導向」。
                    2. 綜合新聞與 PTT 情緒。
                    3. 結合技術指標位置，推算目標價。
                    
                    請以 **純 JSON** 輸出，確保格式正確：
                    {{
                        "sentiment_weight": 70,
                        "weight_reason": "理由...",
                        "chart_data": {{ "target_price": 0, "high_price": 0, "low_price": 0, "buy_price": 0, "sell_price": 0 }},
                        "analysis_report": "## Markdown 報告內容..."
                    }}
                    """
                    response = model.generate_content(prompt)
                    
                    # 使用 Regex 清理 AI 回傳的 Markdown 符號 (```json ... ```)
                    clean_text = re.sub(r'```json|```', '', response.text).strip()
                    ai_data = json.loads(clean_text)
                    
                    # 顯示動態權重條 (藍色進度條)
                    if 'sentiment_weight' in ai_data:
                        w = ai_data['sentiment_weight']
                        st.info(f"⚖️ 消息權重: {w}% (Beta校正) | 技術權重: {100-w}%")
                        st.progress(w/100)
                    
                    # 顯示文字報告
                    if 'analysis_report' in ai_data:
                        st.markdown(ai_data['analysis_report'])
                        
                    # 在圖表上畫預測線 (從今天連到明天)
                    if 'chart_data' in ai_data:
                        c = ai_data['chart_data']
                        now = datetime.now()
                        # 計算畫線的起點 (如果是週末或資料延遲，從今天開始畫)
                        start_pt = now if (now - last_date).days > 1 else last_date
                        # 計算明天 (跳過週末)
                        next_dt = now + timedelta(days=1)
                        while next_dt.weekday() > 4: next_dt += timedelta(days=1)
                        
                        # 加三條線到圖表上
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('high_price', last_close)], mode='lines+markers', line=dict(color='red', dash='dot'), name='樂觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('low_price', last_close)], mode='lines+markers', line=dict(color='green', dash='dot'), name='悲觀'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[start_pt, next_dt], y=[last_close, c.get('target_price', last_close)], mode='lines+markers', line=dict(color='orange', width=4), name='目標'), row=1, col=1)

                except Exception as e:
                    st.error(f"AI 分析失敗: {e}")
        
        # 畫出最終圖表
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # 分頁 2: 蒙地卡羅風險模擬
    # ==========================
    with tab2:
        st.header("🎲 蒙地卡羅風險模擬 (Monte Carlo Simulation)")
        
        # ────────── 這段就是你想要的「易懂解釋」──────────
        st.info("""
        **為什麼模擬結果長這樣？（用白話解釋）**
        1. **所有線都從今天股價開始** → 因為我們不知道明天會漲還是跌，只能從「現在」出發。
        2. **淡藍色線像扇子一樣越張越開** → 時間越久，未來越不確定！  
           就像丟一顆骰子：丟一次大概知道範圍，丟 100 次就什麼結果都可能發生。
        3. **橘色粗線是「平均預期」** → 這 500 次模擬的平均結果，代表「最可能的長期走勢」  
           （會微微往上或往下，是因為這檔股票過去真的有這種趨勢）
        4. **95% VaR 是「最慘的 5% 情況」** → 500 次模擬裡，挑出最慘的前 25 次，算平均。  
           這就是銀行、金控、交易員每天在看的「極端風險值」！
        5. **紅黃綠燈是實務風控標準** • 紅燈（>15%）→ 老闆看到會叫你停損  
           • 黃燈（8~15%）→ 要設停損點了  
           • 綠燈（<8%）→ 可以安心抱著睡
        簡單說：這不是亂畫的線，而是**用 500 個平行宇宙幫你預演未來**！
        """)
        # ───────────────────────────────────────
        st.caption("基於幾何布朗運動 (GBM) 模型，模擬未來股價路徑與風險值 (VaR)。")
        
        # 初始化 session state 變數，防止切換時報錯
        if 'run_mc' not in st.session_state:
            st.session_state.run_mc = False
        
        mc_col1, mc_col2 = st.columns([1, 3])
        
        try:
            # 呼叫快取函式計算參數
            log_returns, daily_volatility, drift, annual_volatility = calculate_metrics(df)
        except Exception as e:
            st.error(f"指標計算錯誤: {e}")
            st.stop()

        with mc_col1:
            st.subheader("參數設定")
            sim_days = st.slider("模擬天數", 30, 365, 90)
            n_simulations = st.slider("模擬次數", 100, 1000, 500)
            initial_investment = st.number_input("投資金額", value=100000, step=10000)
            st.metric("年化波動率", f"{annual_volatility*100:.2f}%")

        with mc_col2:
            col_btn, col_clear = st.columns([1, 4])
            with col_btn:
                # 按下按鈕，執行運算
                if st.button("🎲 開始模擬運算", type="primary", use_container_width=True):
                    with st.spinner("正在計算..."):
                        last_price = last_close
                        all_paths = []
                        # 跑 n 次迴圈，生成隨機路徑
                        for i in range(n_simulations):
                            # np.random.normal(0, 1) 是產生常態分佈隨機數 (Z-score)
                            daily_shocks = drift + daily_volatility * np.random.normal(0, 1, sim_days)
                            price_paths = [last_price]
                            for shock in daily_shocks:
                                price_paths.append(price_paths[-1] * np.exp(shock))
                            all_paths.append(price_paths)
                        
                        # 畫蒙地卡羅圖
                        fig_mc = go.Figure()
                        x_axis = list(range(sim_days + 1))
                        # 只畫前 100 條，不然網頁會卡死
                        for path in all_paths[:100]:
                            fig_mc.add_trace(go.Scatter(x=x_axis, y=path, mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)', width=1), showlegend=False))
                        
                        # 畫平均線
                        avg_path = np.mean(all_paths, axis=0)
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', line=dict(color='orange', width=3), name='平均預期'))
                        
                        # 計算風險值 (VaR)
                        final_prices = [p[-1] for p in all_paths]
                        expected_return = (np.mean(final_prices) - last_price) / last_price
                        # np.percentile(..., 5) 代表找第 5% 差的那個價格 (最差情況)
                        var_95_price = np.percentile(final_prices, 5)
                        loss_at_risk = (last_price - var_95_price) / last_price
                        
                        # 存起來！ (Session State)
                        st.session_state.mc_fig = fig_mc
                        st.session_state.mc_return = expected_return
                        st.session_state.mc_risk = loss_at_risk
                        st.session_state.mc_asset = initial_investment * (1-loss_at_risk)
                        st.session_state.run_mc = True

            # 如果以前跑過，就顯示結果 (持久化顯示)
            if st.session_state.run_mc and 'mc_fig' in st.session_state:
                st.plotly_chart(st.session_state.mc_fig, use_container_width=True)
                
                r1, r2, r3 = st.columns(3)
                r1.metric("預期報酬率", f"{st.session_state.mc_return*100:.2f}%")
                r2.metric("95% VaR 風險值", f"-{st.session_state.mc_risk*100:.2f}%")
                
                # 風險警報系統 (Alert System)
                risk = st.session_state.mc_risk
                if risk > 0.15:
                    st.error("🚨 高風險警報：虧損可能超過 15%！")
                elif risk > 0.08:
                    st.warning("⚠️ 中度風險：建議設停損。")
                else:
                    st.success("✅ 低風險區域。")
            
            with col_clear:
                if st.session_state.run_mc:
                    if st.button("清除模擬"):
                        st.session_state.run_mc = False
                        st.rerun() # 重新整理網頁
