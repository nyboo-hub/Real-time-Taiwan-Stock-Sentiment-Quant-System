import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from GoogleNews import GoogleNews
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import re
import twstock

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 智能台股情緒量化分析系統", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)** 與 **Generative AI (預測模型)** 的雙軌決策系統。  
> **技術架構**：Python ETL Pipeline + Google Gemini LLM + Streamlit Cloud
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
        
        priorities = ['gemma-3n-e4b-it', 'gemini-2.5-pro-preview-03-25', 'gemini-1.5-flash']
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

# --- 5. 主程式邏輯 ---

if st.button("🚀 啟動全方位分析"):
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。請在側邊欄輸入或檢查 Secrets 設定。")
        st.stop()

    # --- A. 量化分析 ---
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"找不到 {ticker} 的股價資料，請確認代號是否正確。")
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

        last_date = df.index[-1]
        last_close = float(df['Close'].iloc[-1])
        last_change = last_close - float(df['Close'].iloc[-2])
        last_rsi = df['RSI'].iloc[-1]
        
        ma5_val = df['MA5'].iloc[-1]
        ma20_val = df['MA20'].iloc[-1]
        upper_val = df['Upper'].iloc[-1]
        lower_val = df['Lower'].iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}", f"{last_change:.2f}")
        c2.metric("MA20 (月線)", f"{ma20_val:.2f}")
        c3.metric("波動率 (STD)", f"{df['STD'].iloc[-1]:.2f}")
        c4.metric("RSI (14)", f"{last_rsi:.2f}")

    except Exception as e:
        st.error(f"數據分析發生錯誤: {e}")
        st.stop()

    # --- B. 質化分析 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{stock_name} 價格走勢與 AI 預測', 'RSI 強弱指標'),
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='歷史K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='布林通道'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    st.markdown("---")
    st.header(f"📰 AI 華爾街分析師報告 (Model: {selected_model_name})")
    
    col_news, col_ai = st.columns([1, 1])
    
    news_text_for_ai = ""
    
    with col_news:
        st.subheader("即時新聞爬蟲")
        with st.spinner("正在爬取 Google News..."):
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                news_result = googlenews.result()[:5] 
                
                if news_result:
                    for item in news_result:
                        st.markdown(f"- **{item['title']}**")
                        st.caption(f"來源: {item['media']} | 時間: {item['date']}")
                        news_text_for_ai += f"{item['title']} (來源: {item['media']})\n"
                else:
                    st.warning("找不到近期新聞")
                    news_text_for_ai = "查無近期特定新聞，請基於市場一般認知進行分析。"
            except Exception as e:
                st.error(f"新聞爬蟲失敗: {e}")
                news_text_for_ai = "新聞抓取失敗。"

    with col_ai:
        st.subheader("Gemini 雙軌投資決策")
        with st.spinner(f"正在連線 {selected_model_name} 進行深度推演..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(selected_model_name)
                
                today_str = datetime.now().strftime("%Y年%m月%d日")
                
                prompt = f"""
                你是一位專業的華爾街量化交易員。
                今天是 **{today_str}**。
                目前 {stock_name} ({ticker}) 的**最新收盤價為 {last_close:.2f} 元**。
                
                請根據以下新聞與技術指標進行分析：
                {news_text_for_ai}
                技術指標：RSI={last_rsi:.2f}, MA20={ma20_val:.2f}, MA5={ma5_val:.2f}, 布林上軌={upper_val:.2f}, 布林下軌={lower_val:.2f}
                
                請以 **純 JSON 格式** 輸出，嚴格遵守 JSON 規範。
                JSON 結構範例：
                {{
                    "chart_data": {{ "target_price": 1050.5, "buy_price": 1030, "sell_price": 1080 }},
                    "analysis_report": "## 標題\\n\\n報告內容..."
                }}
                """
                
                response = model.generate_content(prompt)
                raw_text = response.text
                clean_text = re.sub(r'```json|```', '', raw_text).strip()
                
                ai_data = None
                try:
                    ai_data = json.loads(clean_text, strict=False)
                except:
                    st.warning("⚠️ 格式解析啟動相容模式")
                    # Regex Fallback
                    tp_match = re.search(r'"target_price":\s*([\d\.]+)', clean_text)
                    bp_match = re.search(r'"buy_price":\s*([\d\.]+)', clean_text)
                    sp_match = re.search(r'"sell_price":\s*([\d\.]+)', clean_text)
                    
                    ai_data = {
                        "chart_data": {
                            "target_price": float(tp_match.group(1)) if tp_match else last_close,
                            "buy_price": float(bp_match.group(1)) if bp_match else last_close * 0.95,
                            "sell_price": float(sp_match.group(1)) if sp_match else last_close * 1.05
                        },
                        "analysis_report": raw_text 
                    }

                if "analysis_report" in ai_data:
                    st.markdown(ai_data['analysis_report'])
                else:
                    st.markdown(raw_text)
                
                if 'chart_data' in ai_data:
                    chart_data = ai_data['chart_data']
                    raw_target = chart_data['target_price']
                    
                    # --- 關鍵修正：Python 邏輯熔斷機制 (Circuit Breaker) ---
                    # 台灣股市漲跌幅限制為 10%
                    limit_up = last_close * 1.10
                    limit_down = last_close * 0.90
                    
                    final_target = raw_target
                    
                    # 檢查是否超漲或超跌
                    is_crazy = False
                    if final_target > limit_up:
                        final_target = limit_up
                        is_crazy = True
                        st.warning(f"⚠️ AI 原始預測 ({raw_target:.2f}) 超出台股漲幅限制，系統已自動修正為漲停價 ({final_target:.2f})。")
                    elif final_target < limit_down:
                        final_target = limit_down
                        is_crazy = True
                        st.warning(f"⚠️ AI 原始預測 ({raw_target:.2f}) 超出台股跌幅限制，系統已自動修正為跌停價 ({final_target:.2f})。")
                    
                    # --- 日期修正：確保畫在明天 ---
                    now = datetime.now()
                    # 如果資料庫最後日期跟今天差太多(超過3天)，我們就從「今天」開始畫
                    start_point_date = last_date
                    start_point_price = last_close
                    
                    if (now - last_date).days > 1:
                        # 補一個今天的點，讓線連起來
                        start_point_date = now
                    
                    next_date = now + timedelta(days=1)
                    while next_date.weekday() > 4: # 避開週末
                        next_date += timedelta(days=1)
                    
                    # 畫出預測線
                    fig.add_trace(go.Scatter(
                        x=[start_point_date, next_date],
                        y=[start_point_price, final_target],
                        mode="lines+markers",
                        line=dict(color="red", width=3, dash="dot"),
                        name=f"AI 預測 ({final_target:.2f})"
                    ), row=1, col=1)
                    
                    # 買賣點
                    fig.add_hline(y=chart_data['buy_price'], line_dash="dash", line_color="green", annotation_text="買進", row=1, col=1)
                    fig.add_hline(y=chart_data['sell_price'], line_dash="dash", line_color="red", annotation_text="賣出", row=1, col=1)

            except Exception as e:
                st.error(f"分析錯誤: {e}")

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
