import pandas as pd
import yfinance as yf
import xgboost as xgb
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"

data = yf.download(ticker,start = start_date,end=end_date)

df = pd.DataFrame(data)

#use the date as the index
df['date']=pd.to_datetime(df.index)

#candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
#Meta chart info
fig.update_layout(
    title = 'Stock Pric AAPL',
    yaxis_title='Price($)',
    xaxis_rangeslider_visible=False)
#fig.show()

df.drop(['date','Volume'],axis=1,inplace=True)
df.reset_index(drop=True, inplace=True)
df.plot.line(y='Close',use_index=True)

x = df[['Open','Close', 'High', 'Low', 'Adj Close']]
y = df[['Close']] #to predict
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train,Y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)
print("MSE = ",mse)

