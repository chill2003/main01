import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import twstock
import random
import plost
import mplfinance as mpf



model = load_model('C:/Users/User/OneDrive/桌面/Stock Predictions Model.keras')
st.set_page_config(layout = 'wide',initial_sidebar_state='expanded')

st.sidebar.header('金融APP')

# 台股資料
data_load_state = st.text("請輸入台股代碼!並按下Enter鍵!")
st.sidebar.header('台灣股票')
stock_tw = st.sidebar.text_input('請輸入台股代碼', '2330')  # 以台積電股票為例
data_tw = twstock.Stock(stock_tw).fetch_from(2022, 1)
data_load_state.text("資料加載完成!!")

#今日最高價和最低價
col1, col2, col3 = st.columns(3)

# 获取最后一天和倒数第二天的数据
latest_data = data_tw[-1]
second_latest_data = data_tw[-2]

if latest_data and second_latest_data:
    # 计算涨跌百分比
    high_change = ((latest_data.high - second_latest_data.high) / second_latest_data.high) * 100
    low_change = ((latest_data.low - second_latest_data.low) / second_latest_data.low) * 100

    # 显示最新数据
    col1.metric("今日最高價", f"{latest_data.high}",f"{high_change:.2f}%")
    col2.metric("今日最低價", f"{latest_data.low}",f"{low_change:.2f}%")
else:
    st.write("無數據，無法顯示最新的最高價和最低價以及相應的涨跌幅。")

# 將台股資料轉換為 DataFrame 格式
data_tw_df = pd.DataFrame({
    'date': [data.date for data in data_tw],
    '收盤價': [data.close for data in data_tw],
    '開盤價': [data.open for data in data_tw],
    '當日最高價': [data.high for data in data_tw],
    '當日最低價': [data.low for data in data_tw],
    '總交易金額': [data.capacity for data in data_tw],
    '成交量(股)': [data.turnover for data in data_tw]
})

st.subheader(f"{stock_tw}股票資料")
st.write(data_tw_df)
# 台股資料預處理
data_tw_df = pd.DataFrame({
    'date': [data.date for data in data_tw],
    'close': [data.close for data in data_tw],
    'open': [data.open for data in data_tw],
    'high': [data.high for data in data_tw],
    'low': [data.low for data in data_tw],
    'volume': [data.capacity for data in data_tw],
    'turnover': [data.turnover for data in data_tw]
})

data_train_tw = pd.DataFrame(data_tw_df.close[0: int(len(data_tw_df)*0.80)])
data_test_tw = pd.DataFrame(data_tw_df.close[int(len(data_tw_df)*0.80): len(data_tw_df)])

from sklearn.preprocessing import MinMaxScaler
scaler_tw = MinMaxScaler(feature_range=(0,1))

past_100_days_tw = data_train_tw.tail(100)
data_test_tw = pd.concat([past_100_days_tw, data_test_tw], ignore_index=True)
data_test_scale_tw = scaler_tw.fit_transform(data_test_tw)


# 计算移动平均线
#MA50
ma_50_days_tw = data_tw_df.close.rolling(50).mean()

# 將日期欄位設置為索引並轉換為 DatetimeIndex
data_tw_df['date'] = pd.to_datetime(data_tw_df['date'])
data_tw_df.set_index('date', inplace=True)

# 原始交易價格 vs MA50 vs MA100
st.subheader('原始交易價格 vs MA50 vs MA100')
ma_100_days_tw = data_tw_df.close.rolling(100).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(data_tw_df.index, ma_50_days_tw, 'r', label='MA50')
plt.plot(data_tw_df.index, ma_100_days_tw, 'b', label='MA100')
plt.plot(data_tw_df.index, data_tw_df.close, 'g', label='Price')
plt.legend()  # 添加圖例
plt.xlabel('Time')  # 设置 x 轴标签
plt.ylabel('Price')  # 设置 y 轴标签
plt.xticks(rotation=45)  # 旋转 x 轴标签，以免重叠
st.pyplot(fig1)

# 計算 MACD
data_tw_df['12EMA'] = data_tw_df['close'].ewm(span=12, adjust=False).mean()
data_tw_df['26EMA'] = data_tw_df['close'].ewm(span=26, adjust=False).mean()
data_tw_df['MACD'] = data_tw_df['12EMA'] - data_tw_df['26EMA']
data_tw_df['Signal'] = data_tw_df['MACD'].ewm(span=9, adjust=False).mean()
data_tw_df['Histogram'] = data_tw_df['MACD'] - data_tw_df['Signal']

# 繪製 MACD 圖
st.subheader("MACD 圖")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data_tw_df.index, data_tw_df['MACD'], label='MACD', color='blue')
ax.plot(data_tw_df.index, data_tw_df['Signal'], label='Signal Line', color='red')
ax.bar(data_tw_df.index, data_tw_df['Histogram'], width=0.05, label='Histogram', color='green')
ax.legend()
st.pyplot(fig)

# 台股資料預測
x_tw = []
y_tw = []

for i in range(100, data_test_scale_tw.shape[0]):
    x_tw.append(data_test_scale_tw[i-100:i])
    y_tw.append(data_test_scale_tw[i,0])

x_tw, y_tw = np.array(x_tw), np.array(y_tw)

predict_tw = model.predict(x_tw)

scale_tw = 1/scaler_tw.scale_

predict_tw = predict_tw * scale_tw
y_tw = y_tw * scale_tw

st.subheader('原始價格 vs 預估價格 (台灣股市)')
fig2 = plt.figure(figsize=(8,6))
plt.plot(predict_tw, 'r', label='Original Price')
plt.plot(y_tw, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
