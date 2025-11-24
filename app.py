import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from GoogleNews import GoogleNews
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 智能台股情緒量化分析系統", layout="wide")
st.title("📈 AI 智能台股情緒量化分析系統")
st.markdown("""
> **專案亮點**：結合 **統計學 (MA/布林通道)** 與 **Generative AI (SOTA Model)** 的雙軌決策系統。  
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
selected_model_name = "gemini-1.5-flash"

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
        
        if "preview" in selected_model_name:
            st.sidebar.success(f"🚀 已啟用最新預覽版: {selected_model_name}")
        elif "flash" in selected_model_name:
            st.sidebar.info(f"⚡ 已啟用高速推論模式")
            
    except Exception as e:
        st.sidebar.error(f"連線錯誤，將使用預設模型")

# --- 4. 股票參數設定 (智慧連動版) ---
st.sidebar.header("📊 股票參數")

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
    code = input_val.split('.')[0]
    if code in TW_STOCK_MAP:
        st.session_state.stock_name_input = TW_STOCK_MAP[code]

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

        # 取得最新數據供 AI 參考
        last_close = df['Close'].iloc[-1]
        last_change = last_close - df['Close'].iloc[-2]
        
        # 提取技術指標數值 (準備餵給 AI)
        ma5_val = df['MA5'].iloc[-1]
        ma20_val = df['MA20'].iloc[-1]
        upper_val = df['Upper'].iloc[-1]
        lower_val = df['Lower'].iloc[-1]
        
        # 顯示關鍵指標
        c1, c2, c3 = st.columns(3)
        c1.metric("最新收盤價", f"{last_close:.2f}", f"{last_change:.2f}")
        c2.metric("MA20 (月線)", f"{ma20_val:.2f}")
        c3.metric("標準差 (波動率)", f"{df['STD'].iloc[-1]:.2f}")

        # 繪製圖表
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='布林通道'))
        fig.update_layout(title=f"{stock_name} ({ticker}) 股價走勢圖", xaxis_title="日期", yaxis_title="價格", height=500)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"數據分析發生錯誤: {e}")
        st.stop()

    # --- B. 質化分析 (Qualitative Analysis via AI) ---
    st.markdown("---")
    st.header(f"📰 AI 華爾街分析師報告 (Model: {selected_model_name})")
    
    col_news, col_ai = st.columns([1, 1])
    
    with col_news:
        st.subheader("即時新聞爬蟲")
        with st.spinner("正在爬取 Google News..."):
            try:
                googlenews = GoogleNews(lang='zh-TW', region='TW')
                googlenews.search(stock_name)
                news_result = googlenews.result()[:5] 
                
                news_text_for_ai = ""
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
                
                # --- 關鍵修改：取得當前日期並強制寫入 Prompt ---
                today_str = datetime.now().strftime("%Y年%m月%d日")
                
                prompt = f"""
                你是一位專業的華爾街量化交易員。請進行深度的**雙軌分析（技術面 + 消息面）**。
                
                **⚠️ 重要：今天是 {today_str}。所有分析請基於此日期，若新聞是舊的請在分析中註明。**
                
                ### 【市場數據輸入】
                * **股票名稱**：{stock_name} ({ticker})
                * **最新收盤價**：{last_close:.2f} 元
                * **技術指標**：
                    - MA5 (週線)：{ma5_val:.2f}
                    - MA20 (月線)：{ma20_val:.2f}
                    - 布林通道上軌：{upper_val:.2f}
                    - 布林通道下軌：{lower_val:.2f}
                
                ### 【新聞消息輸入】
                {news_text_for_ai}
                
                ---
                
                請用繁體中文輸出以下結構化分析報告，**數值請給出具體數字**：
                **報告標題：{stock_name} ({ticker}) 雙軌分析報告 - {today_str}**

                ### 1. 🏛️ 技術面分析 (Technical Analysis)
                * **趨勢判斷**：請基於 MA5 與 MA20 的位置（例如黃金交叉/死亡交叉），以及目前股價在布林通道的位置，判斷目前是多頭、空頭還是盤整。
                * **支撐與壓力**：利用上述技術指標數值，指出目前的強力支撐位與壓力位。
                
                ### 2. 📰 市場情緒分析 (Sentiment Analysis)
                * **新聞情緒評分**：(0~10分，10分為極度樂觀，0分為極度悲觀)
                * **情緒解讀**：分析新聞背後的市場心理 (例如：雖然營收好但利多出盡...)。
                
                ### 3. 🔮 AI 價格預測 (明日)
                * **上漲機率**：______ % (0-100%)
                * **預估漲跌幅**：______ % (例如 +1.2%，請考慮台股 10% 限制)
                * **預估收盤價**：______ 元
                
                ### 4. ♟️ 交易策略建議
                * **🎯 建議買進價**：______ 元
                * **🚀 建議賣出價**：______ 元
                * **綜合點評**：一句話總結技術面與消息面的綜合判斷。
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
                st.success("分析完成！數據僅供學術研究參考。")
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error(f"❌ 配額不足 (429)：模型 {selected_model_name} 目前忙碌，請在左側切換回 gemini-1.5-flash 再試一次。")
                elif "404" in error_msg:
                    st.error(f"❌ 模型未找到 (404)：您的 API 帳號可能無法使用 `{selected_model_name}`，或是名稱打錯了。請在下拉選單換回 gemini-1.5-flash。")
                else:
                    st.error(f"❌ 分析失敗: {e}")