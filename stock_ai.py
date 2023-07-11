from datetime import datetime, timedelta
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

USERNAME="stocks"
PASSWORD="stocks"

connection = mysql.connector.connect(host='localhost',
                                    database='stocks',
                                    user=USERNAME,
                                    password=PASSWORD)

cursor = connection.cursor()

class stock_ai:
    def __init__(self, ticker, end_date, num_days):
        self.ticker = ticker
        start_date = end_date - timedelta(days=num_days)

        sql_query = f"SELECT * FROM historical_data WHERE symbol = '{self.ticker}' AND '{start_date}' <= Date AND Date <= '{end_date}' ORDER BY Date ASC"
        cursor.execute(sql_query)

        self.df_sql_data = pd.DataFrame.from_records(cursor.fetchall(), columns=[x[0] for x in cursor.description])


    def fill_deltas(self):

        self.df_sql_data['Date'] = pd.to_datetime(self.df_sql_data['Date'])
        self.df_sql_data = self.df_sql_data.set_index('Date')

        self.df_sql_data["daily_volatility"] = (self.df_sql_data["high"] - self.df_sql_data["low"]) / ((self.df_sql_data["high"] + self.df_sql_data["low"]) / 2)

        self.df_sql_data["daily_volatility"]  = self.df_sql_data["daily_volatility"]  * 100.00

        self.df_sql_data['d_high'] = (self.df_sql_data['high'] - self.df_sql_data['open']) / self.df_sql_data['open']

        self.df_sql_data['d_low'] = (self.df_sql_data['low'] - self.df_sql_data['open']) / self.df_sql_data['open']

        self.df_sql_data['d_close'] = (self.df_sql_data['close'] - self.df_sql_data['open']) / self.df_sql_data['open']

        self.df_sql_data['high5'] = self.df_sql_data['high'].rolling(5).max()

        self.df_sql_data['low5'] = self.df_sql_data['low'].rolling(5).min()

        self.df_sql_data['open5'] = self.df_sql_data['open'].shift(4)

        self.df_sql_data['close5'] = self.df_sql_data['close'].shift(4)

        self.df_sql_data['high10'] = self.df_sql_data['high'].rolling(10).max()

        self.df_sql_data['low10'] = self.df_sql_data['low'].rolling(10).min()

        self.df_sql_data['open10'] = self.df_sql_data['open'].shift(9)

        self.df_sql_data['close10'] = self.df_sql_data['close'].shift(9)

        self.df_sql_data['high30'] = self.df_sql_data['high'].rolling(30).max()

        self.df_sql_data['low30'] = self.df_sql_data['low'].rolling(30).min()

        self.df_sql_data['open30'] = self.df_sql_data['open'].shift(29)

        self.df_sql_data['close30'] = self.df_sql_data['close'].shift(29)


        self.df_sql_data['d_high5'] = (self.df_sql_data['high5'] - self.df_sql_data['open5']) / self.df_sql_data['open5']

        self.df_sql_data['d_low5'] = (self.df_sql_data['low5'] - self.df_sql_data['open5']) / self.df_sql_data['open5']

        self.df_sql_data['d_close5'] = (self.df_sql_data['close5'] - self.df_sql_data['open5']) / self.df_sql_data['open5']

        self.df_sql_data['d_high10'] = (self.df_sql_data['high10'] - self.df_sql_data['open10']) / self.df_sql_data['open10']

        self.df_sql_data['d_low10'] = (self.df_sql_data['low10'] - self.df_sql_data['open10']) / self.df_sql_data['open10']

        self.df_sql_data['d_close10'] = (self.df_sql_data['close10'] - self.df_sql_data['open10']) / self.df_sql_data['open10']

        self.df_sql_data['d_high30'] = (self.df_sql_data['high30'] - self.df_sql_data['open30']) / self.df_sql_data['open30']

        self.df_sql_data['d_low30'] = (self.df_sql_data['low30'] - self.df_sql_data['open30']) / self.df_sql_data['open30']

        self.df_sql_data['d_close30'] = (self.df_sql_data['close30'] - self.df_sql_data['open30']) / self.df_sql_data['open30']

        self.df_sql_data['high_5next'] =  self.df_sql_data['high'].rolling(5).max().shift(-5)
        self.df_sql_data['low_5next'] =  self.df_sql_data['low'].rolling(5).min().shift(-5)

        self.df_sql_data['win5'] =  ((self.df_sql_data['high_5next'] - self.df_sql_data["open"]) /  self.df_sql_data["open"]) * 100
        self.df_sql_data['loss5'] =  ((self.df_sql_data['low_5next'] - self.df_sql_data["open"]) /  self.df_sql_data["open"]) * 100

        self.df_sql_data['WIN']  = np.where(self.df_sql_data['win5'] > 6, 'buy', '')

        price_change = self.df_sql_data['close'].diff()

        gain = price_change.where(price_change > 0, 0)
        loss = -price_change.where(price_change < 0, 0)

        AG = gain.rolling(window=14).mean()
        AL = loss.rolling(window=14).mean()

        RS = AG / AL

        self.df_sql_data['RSI'] = 100 - (100 / (1 + RS))

        self.df_sql_data['SMA'] = self.df_sql_data['close'].rolling(window=10).mean()

        self.df_sql_data['SMA5'] = self.df_sql_data['close'].rolling(window=5).mean()

        self.df_sql_data['SMA30'] = self.df_sql_data['close'].rolling(window=30).mean()

        self.df_sql_data['d_SMA'] = (self.df_sql_data['SMA']-self.df_sql_data['close'])/self.df_sql_data['close']

        self.df_sql_data['d_SMA5'] = (self.df_sql_data['SMA5']-self.df_sql_data['close'])/self.df_sql_data['close']

        self.df_sql_data['d_SMA30'] = (self.df_sql_data['SMA30']-self.df_sql_data['close'])/self.df_sql_data['close']

        #print(self.df_sql_data)
        #self.df_sql_data.to_excel('stock_h.xlsx', index=True)


        #print("Dataframe from SQL")


        #''' 
        #mpf.plot(self.df_sql_data, type='candle', title='Price Graph with Volume and Volatility', ylabel='Price',
            #addplot=[mpf.make_addplot([volatility] * len(self.df_sql_data), panel=1, color='orange', linestyle='--', ylabel='Volatility')],
            #volume=True)
        #'''

        input_columns = ['d_high', 'd_low', 'd_close', 'volume', 'd_high5', 'd_low5', 'd_close5', 'd_high10', 'd_low10', 'd_close10', 'd_high30', 'd_low30', 'd_close30', 'RSI', 'd_SMA', 'd_SMA5']

        self.df_sql_data = self.df_sql_data.dropna()

        self.x = self.df_sql_data[input_columns].values

        self.y = self.df_sql_data['WIN'].values
    

    def train(self):
        split_index = int(len(self.x) * 0.7)

        x_train = self.x[:split_index]
        y_train = self.y[:split_index]

        x_test = self.x[split_index:]
        global y_test
        y_test = self.y[split_index:]

        y_train = np.where(y_train == 'buy', 1, 0)
        y_test = np.where(y_test == 'buy', 1, 0)
        print("------- Y ------ ")
        print(y_train)

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        global x_test_scaled
        x_test_scaled = scaler.transform(x_test)

        global model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train_scaled, y_train, epochs=5000, batch_size=320)

    def test(self):
        results = model.evaluate(x_test_scaled, y_test)

        loss = results[0]
        accuracy = results[1]

        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

google_stock = stock_ai('GOOGL', datetime(2023, 7, 10), 3650)

google_stock.fill_deltas()

google_stock.train()

google_stock.test()

cursor.close()
connection.close()
