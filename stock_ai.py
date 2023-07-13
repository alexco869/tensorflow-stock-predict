from datetime import datetime, timedelta
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

USERNAME="stocks"
PASSWORD="stocks"

class stock_ai:
    
    def __init__(self, ticker, end_date, num_days):
        self.connection = mysql.connector.connect(host='localhost',
                                    database='stocks',
                                    user=USERNAME,
                                    password=PASSWORD)
        
        self.cursor = self.connection.cursor()
        self.end_date = end_date
        self.ticker = ticker
        self.start_date = end_date - timedelta(days=num_days)
        self.input_columns = ['d_high', 'd_low', 'd_close', 'volume', 'd_high5', 'd_low5', 'd_close5', 'd_high10', 'd_low10', 'd_close10', 'd_high30', 'd_low30', 'd_close30', 'RSI', 'd_SMA', 'd_SMA5']
        self.scaler = MinMaxScaler()
        self.output_columns = ['WIN', 'LOSS', 'STRONGWIN']

        sql_query = f"SELECT * FROM historical_data WHERE symbol = '{self.ticker}' AND '{self.start_date}' <= Date AND Date <= '{self.end_date}' ORDER BY Date ASC"
        self.cursor.execute(sql_query)

        self.df_sql_data = pd.DataFrame.from_records(self.cursor.fetchall(), columns=[x[0] for x in self.cursor.description])

    def __del__(self):
        self.cursor.close()
        self.connection.close()

    def update(self):
        query = f"delete FROM historical_data WHERE symbol = '{self.ticker}'"
        self.cursor.execute(query)
        self.connection.commit()
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print("GO to insert ",len(df))
        for _, row in df.iterrows():
            query2 = "INSERT INTO historical_data (Date, open, high, low, close, adjClose, volume, symbol) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            values = (row.name, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'], self.ticker)
            self.cursor.execute(query2, values)
            self.connection.commit()

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

        self.df_sql_data['STRONGWIN']  = np.where(self.df_sql_data['win5'] > 10, 'strong buy', '')

        self.df_sql_data['LOSS']  = np.where(self.df_sql_data['loss5'] > 5, 'sell', '')

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
        filename = 'stock_'+self.ticker+ '.xlsx'
        print("SAVING FILE ",filename)
        self.df_sql_data.to_excel(filename, index=True)


        #print("Dataframe from SQL")


        #''' 
        #mpf.plot(self.df_sql_data, type='candle', title='Price Graph with Volume and Volatility', ylabel='Price',
            #addplot=[mpf.make_addplot([volatility] * len(self.df_sql_data), panel=1, color='orange', linestyle='--', ylabel='Volatility')],
            #volume=True)
        #'''

        self.df_sql_data = self.df_sql_data.dropna()

       
    

    def train(self, num_epochs):
        self.x = self.df_sql_data[self.input_columns].values
        self.y = self.df_sql_data[self.output_columns].values
        split_index = int(len(self.x) * 0.7)
        x_train = self.x[:split_index]
        x_test = self.x[split_index:]

        y_train_win = np.where(self.df_sql_data['WIN'] == 'buy', 1, 0)
        y_train_strongwin = np.where(self.df_sql_data['STRONGWIN'] == 'strong buy', 1, 0)
        y_train_loss = np.where(self.df_sql_data['LOSS'] == 'sell', 1, 0)

        y_train_win = y_train_win[:split_index]
        y_train_strongwin = y_train_strongwin[:split_index]
        y_train_loss = y_train_loss[:split_index]

        # Pad or truncate the output arrays to match the size of x_train
        y_train_win = np.pad(y_train_win, (0, len(x_train) - len(y_train_win)))
        y_train_strongwin = np.pad(y_train_strongwin, (0, len(x_train) - len(y_train_strongwin)))
        y_train_loss = np.pad(y_train_loss, (0, len(x_train) - len(y_train_loss)))

        y_test_win = np.where(self.df_sql_data['WIN'] == 'buy', 1, 0)[split_index:]
        y_test_strongwin = np.where(self.df_sql_data['STRONGWIN'] == 'strong buy', 1, 0)[split_index:]
        y_test_loss = np.where(self.df_sql_data['LOSS'] == 'sell', 1, 0)[split_index:]

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        model_win = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model_strongwin = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model_loss = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model_win.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_strongwin.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_loss.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        model_win.fit(x_train_scaled, y_train_win, epochs=num_epochs, batch_size=320)
        model_strongwin.fit(x_train_scaled, y_train_strongwin, epochs=num_epochs, batch_size=320)
        model_loss.fit(x_train_scaled, y_train_loss, epochs=num_epochs, batch_size=320)

        results_win = model_win.evaluate(x_test_scaled, y_test_win)
        results_strongwin = model_strongwin.evaluate(x_test_scaled, y_test_strongwin)
        results_loss = model_loss.evaluate(x_test_scaled, y_test_loss)

        loss_win = results_win[0]
        accuracy_win = results_win[1]
        loss_strongwin = results_strongwin[0]
        accuracy_strongwin = results_strongwin[1]
        loss_loss = results_loss[0]
        accuracy_loss = results_loss[1]

        print("Test Loss (WIN):", loss_win)
        print("Test Accuracy (WIN):", accuracy_win)
        print("Test Loss (STRONGWIN):", loss_strongwin)
        print("Test Accuracy (STRONGWIN):", accuracy_strongwin)
        print("Test Loss (LOSS):", loss_loss)
        print("Test Accuracy (LOSS):", accuracy_loss)

        self.model_win = model_win
        self.model_strongwin = model_strongwin
        self.model_loss = model_loss
        self.x_test_scaled = x_test_scaled
        self.y_test_win = y_test_win
        self.y_test_strongwin = y_test_strongwin
        self.y_test_loss = y_test_loss



    def test(self):
        model_win = self.model_win
        model_strongwin = self.model_strongwin
        model_loss = self.model_loss
        x_test_scaled = self.x_test_scaled
        y_test_win = self.y_test_win
        y_test_strongwin =self.y_test_strongwin
        y_test_loss = self.y_test_loss

        results_win = model_win.evaluate(x_test_scaled, y_test_win)
        results_strongwin = model_strongwin.evaluate(x_test_scaled, y_test_strongwin)
        results_loss = model_loss.evaluate(x_test_scaled, y_test_loss)

        loss_win = results_win[0]
        accuracy_win = results_win[1]

        loss_strongwin = results_strongwin[0]
        accuracy_strongwin = results_strongwin[1]

        loss_loss = results_loss[0]
        accuracy_loss = results_loss[1]

        print("Test Loss (WIN):", loss_win)
        print("Test Accuracy (WIN):", accuracy_win)

        print("Test Loss (STRONGWIN):", loss_strongwin)
        print("Test Accuracy (STRONGWIN):", accuracy_strongwin)

        print("Test Loss (LOSS):", loss_loss)
        print("Test Accuracy (LOSS):", accuracy_loss)
 




    def evaluate(self):
        model_win = self.model_win
        model_strongwin = self.model_strongwin
        model_loss = self.model_loss
        
        last_row = self.df_sql_data.tail(1)[self.input_columns]
        last_row_scaled = self.scaler.transform(last_row.values)

        prediction_win = model_win.predict(last_row_scaled)
        prediction_strongwin = model_strongwin.predict(last_row_scaled)
        prediction_loss = model_loss.predict(last_row_scaled)

        if prediction_win[0] >= 0.5:
            print("Buy (WIN)")
        else:
            print("Don't Buy (WIN)")

        print("Prediction (WIN):", prediction_win[0])

        if prediction_strongwin[0] >= 0.5:
            print("Buy (STRONGWIN)")
        else:
            print("Don't Buy (STRONGWIN)")

        print("Prediction (STRONGWIN):", prediction_strongwin[0])

        if prediction_loss[0] >= 0.5:
            print("Sell (LOSS)")
        else:
            print("Don't Sell (LOSS)")

        print("Prediction (LOSS):", prediction_loss[0])