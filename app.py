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

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 智能台股情緒量化分析系統", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道/RSI)** 與 **Generative AI (預測模型)** 的雙軌決策系統。  
> **技術架構**：Python ETL Pipeline + Google Gemini LLM + Streamlit Cloud
""")

# --- 2. 智慧型 API Key 管理 (防呆版) ---
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

# --- 3. 進階模型選擇器 (包含所有指定模型) ---
selected_model_name = "gemini-1.5-flash"

if api_key:
    st.sidebar.header("🤖 AI 模型設定")
    try:
        genai.configure(api_key=api_key)
        
        # 定義模型清單 (包含你指定的 Gemma)
        target_models = [
            'gemma-3n-e4b-it',              # 你指定的特殊模型
            'gemini-2.5-pro-preview-03-25', # 最新預覽版
            'gemini-1.5-pro',               # 邏輯最強
            'gemini-1.5-flash',             # 速度最快
            'gemini-pro'                    # 舊版保底
        ]
        
        # 嘗試抓取 API 實際可用的模型
        try:
            api_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            api_models = []
            
        # 合併清單並去重
        all_options = list(set(target_models + api_models))
        all_options.sort()
        
        # 調整排序：把常用的排在前面
        priorities = ['gemini-2.5-pro-preview-03-25', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemma-3n-e4b-it']
        for p in reversed(priorities):
            if p in all_options:
                all_options.remove(p)
                all_options.insert(0, p)

        selected_model_name = st.sidebar.selectbox("選擇推論模型 (Model)", all_options, index=0)
        
        if "preview" in selected_model_name:
            st.sidebar.success(f"🚀 已啟用最新預覽版: {selected_model_name}")
        elif "flash" in selected_model_name:
            st.sidebar.info(f"⚡ 已啟用高速推論模式")
        elif "gemma" in selected_model_name:
            st.sidebar.warning(f"🧪 已啟用實驗性模型: {selected_model_name}")
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

# --- 4. 股票參數設定 (智慧連動版) ---
st.sidebar.header("📊 股票參數")

# 熱門台股對照表
TW_STOCK_MAP = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電', '2303': '聯電',
    '2881': '富邦金', '2882': '國泰金', '2891': '中信金', '2886': '兆豐金', '2884': '玉山金',
    '2603': '長榮', '2609': '陽明', '2615': '萬海', '2618': '長榮航', '2610': '華航',
    '3008': '大立光', '3034': '聯詠', '2382': '廣達', '3231': '緯創', '2356': '英業達',
    '2376': '技嘉', '2357': '華碩', '2412': '中華電', '3045': '台灣大', '4904': '遠傳',
    '1301': '台塑', '1303': '南亞', '1326': '台化', '6505': '台塑化', '2002': '中鋼',
    '1101': '台泥', '1216': '統一', '2912': '統一超', '2207': '和泰車', '5871': '中租-KY',
    '3711': '日月光投控', '2379': '瑞昱', '3037': '欣興', '2345': '智邦', '6669': '緯穎',
    '1513': '中興電', '1519': '華城', '1504': '東元', '2371': '大同', '6235': '華孚'
}

def update_stock_name():
    input_val = st.session_state.ticker_input.upper().strip()
    # 只取代號部分
    code = input_val.split('.')[0]
    if code in TW_STOCK_MAP:
        st.session_state.stock_name_input = TW_STOCK_MAP[code]

ticker = st.sidebar.text_input("股票代號 (台股請加 .TW)", value="2330.TW", key="ticker_input", on_change=update_stock_name)
stock_name = st.sidebar.text_input("股票名稱 (用於搜尋新聞)", value="台積電", key="stock_name_input")
days = st.sidebar.slider("分析天數範圍", 30, 365, 120)

# 自動防呆：補上 .TW
if ticker.isdigit():
    ticker = f"{ticker}.TW"

# --- 5. 主程式邏輯 ---

if st.button("🚀 啟動全方位分析"):
    if not api_key:
        st.error("❌ 錯誤：未偵測到 API Key。請在側邊欄輸入或檢查 Secrets 設定。")
        st.stop()

    # --- A. 量化分析 (Quantitative Analysis) ---
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"找不到 {ticker} 的股價資料，請確認代號是否正確。")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 統計運算
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (2 * df['STD']) 
        df['Lower'] = df['MA20'] - (2 * df['STD']) 

        # 計算 RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 取得最新數據
        last_date = df.index[-1]
        last_close = df['Close'].iloc[-1]
        last_change = last_close - df['Close'].iloc[-2]
        last_rsi = df['RSI'].iloc[-1]
        
        # 技術指標數值
        ma20_val = df['MA20'].iloc[-1]
        
        # 顯示關鍵指標
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤", f"{last_close:.2f}", f"{last_change:.2f}")
        c2.metric("MA20 (月線)", f"{ma20_val:.2f}")
        c3.metric("波動率 (STD)", f"{df['STD'].iloc[-1]:.2f}")
        c4.metric("RSI (14)", f"{last_rsi:.2f}")

    except Exception as e:
        st.error(f"數據分析發生錯誤: {e}")
        st.stop()

    # --- B. 質化分析與預測 (Qualitative Analysis via AI) ---
    
    # 初始化圖表 (使用 Subplots)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{stock_name} 價格走勢與 AI 預測', 'RSI 強弱指標'),
                        row_width=[0.2, 0.7])

    # 繪製歷史 K 線
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='歷史K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='布林通道'), row=1, col=1)

    # 繪製 RSI
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
                    st.warning("找不到近期新聞，AI 將僅依據歷史數據進行推論。")
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
                
                # Prompt 設計：要求日期準確 + JSON 格式
                prompt = f"""
                你是一位專業的華爾街量化交易員。
                今天是 **{today_str}**。
                目前 {stock_name} ({ticker}) 的**最新收盤價為 {last_close:.2f} 元**。
                
                請根據以下新聞與技術指標進行分析：
                {news_text_for_ai}
                技術指標：RSI={last_rsi:.2f}, MA20={ma20_val:.2f}
                
                請以 **JSON 格式** 輸出分析結果，不要有 Markdown 標記，格式如下：
                {{
                    "sentiment": "利多/利空/中立",
                    "score": 7,
                    "key_points": ["重點1", "重點2", "重點3"],
                    "prediction": {{
                        "prob_up": 65,
                        "price_change_percent": 1.5,
                        "target_price": 1050.5
                    }},
                    "strategy": {{
                        "buy_price": 1030,
                        "sell_price": 1080,
                        "reason": "簡短策略說明"
                    }},
                    "analysis_summary": "這裡寫一段約 100 字的完整綜合分析文字，包含技術面與消息面。"
                }}
                """
                
                response = model.generate_content(prompt)
                
                # JSON 解析與清理
                raw_text = response.text
                clean_text = re.sub(r'```json|```', '', raw_text).strip()
                ai_data = json.loads(clean_text)
                
                # 顯示文字報告
                st.success(f"市場情緒：{ai_data['sentiment']} (評分: {ai_data['score']}/10)")
                st.info(f"💡 策略：{ai_data['strategy']['reason']}")
                st.markdown(f"**綜合分析**：{ai_data['analysis_summary']}")
                
                with st.expander("查看詳細預測數據"):
                    st.json(ai_data)

                # --- 畫出預測線 ---
                next_date = last_date + timedelta(days=1)
                if next_date.weekday() == 5: next_date += timedelta(days=2)
                elif next_date.weekday() == 6: next_date += timedelta(days=1)
                
                predicted_price = ai_data['prediction']['target_price']
                
                # 預測虛線
                fig.add_trace(go.Scatter(
                    x=[last_date, next_date],
                    y=[last_close, predicted_price],
                    mode="lines+markers",
                    line=dict(color="red", width=3, dash="dot"),
                    name=f"AI 預測 ({predicted_price:.2f})"
                ), row=1, col=1)
                
                # 買賣點水平線
                fig.add_hline(y=ai_data['strategy']['buy_price'], line_dash="dash", line_color="green", annotation_text="買進", row=1, col=1)
                fig.add_hline(y=ai_data['strategy']['sell_price'], line_dash="dash", line_color="red", annotation_text="賣出", row=1, col=1)

            except Exception as e:
                st.error(f"AI 分析或 JSON 解析失敗: {e}")
                st.caption("建議：請重試一次，有時候 AI 輸出的格式會跑掉。")

    # 更新圖表
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
