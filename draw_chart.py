import pandas as pd
import mysql.connector
from datetime import date, timedelta
import mplfinance as mpf
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


TICKER = "AAPL"

connection = mysql.connector.connect(host='localhost',
                                    database='stocks',
                                    user='alex',
                                    password='alex2000')

cursor = connection.cursor()

# -----------------------
end_date = date.today()
start_date = end_date - timedelta(days=3650)

sql_query = f"SELECT * FROM historical_data WHERE symbol = 'MSFT' AND '{start_date}' <= Date AND Date <= '{end_date}' ORDER BY Date ASC"
cursor.execute(sql_query)

df_sql_data = pd.DataFrame.from_records(cursor.fetchall(), columns=[x[0] for x in cursor.description])

# Calcular los retornos logarítmicos
df_sql_data['close'] = pd.to_numeric(df_sql_data['close'])
df_sql_data['Daily_Return'] = df_sql_data['close'].pct_change()

# Calcular la volatilidad utilizando la desviación estándar de los retornos
volatility = df_sql_data['Daily_Return'].std()
print(" Volatility",volatility )

df_sql_data['Date'] = pd.to_datetime(df_sql_data['Date'])
df_sql_data = df_sql_data.set_index('Date')

df_sql_data["daily_volatility"] = (df_sql_data["high"] - df_sql_data["low"]) / ((df_sql_data["high"] + df_sql_data["low"]) / 2)

df_sql_data["daily_volatility"]  = df_sql_data["daily_volatility"]  * 100.00

df_sql_data['d_high'] = (df_sql_data['high'] - df_sql_data['open']) / df_sql_data['open']

df_sql_data['d_low'] = (df_sql_data['low'] - df_sql_data['open']) / df_sql_data['open']

df_sql_data['d_close'] = (df_sql_data['close'] - df_sql_data['open']) / df_sql_data['open']

df_sql_data['high5'] = df_sql_data['high'].rolling(5).max()

df_sql_data['low5'] = df_sql_data['low'].rolling(5).min()

df_sql_data['open5'] = df_sql_data['open'].shift(4)

df_sql_data['close5'] = df_sql_data['close'].shift(4)

df_sql_data['high10'] = df_sql_data['high'].rolling(10).max()

df_sql_data['low10'] = df_sql_data['low'].rolling(10).min()

df_sql_data['open10'] = df_sql_data['open'].shift(9)

df_sql_data['close10'] = df_sql_data['close'].shift(9)

df_sql_data['high30'] = df_sql_data['high'].rolling(30).max()

df_sql_data['low30'] = df_sql_data['low'].rolling(30).min()

df_sql_data['open30'] = df_sql_data['open'].shift(29)

df_sql_data['close30'] = df_sql_data['close'].shift(29)


df_sql_data['d_high5'] = (df_sql_data['high5'] - df_sql_data['open5']) / df_sql_data['open5']

df_sql_data['d_low5'] = (df_sql_data['low5'] - df_sql_data['open5']) / df_sql_data['open5']

df_sql_data['d_close5'] = (df_sql_data['close5'] - df_sql_data['open5']) / df_sql_data['open5']

df_sql_data['d_high10'] = (df_sql_data['high10'] - df_sql_data['open10']) / df_sql_data['open10']

df_sql_data['d_low10'] = (df_sql_data['low10'] - df_sql_data['open10']) / df_sql_data['open10']

df_sql_data['d_close10'] = (df_sql_data['close10'] - df_sql_data['open10']) / df_sql_data['open10']

df_sql_data['d_high30'] = (df_sql_data['high30'] - df_sql_data['open30']) / df_sql_data['open30']

df_sql_data['d_low30'] = (df_sql_data['low30'] - df_sql_data['open30']) / df_sql_data['open30']

df_sql_data['d_close30'] = (df_sql_data['close30'] - df_sql_data['open30']) / df_sql_data['open30']

df_sql_data['high_5next'] =  df_sql_data['high'].rolling(5).max().shift(-5)
df_sql_data['low_5next'] =  df_sql_data['low'].rolling(5).min().shift(-5)

df_sql_data['win5'] =  ((df_sql_data['high_5next'] - df_sql_data["open"]) /  df_sql_data["open"]) * 100
df_sql_data['loss5'] =  ((df_sql_data['low_5next'] - df_sql_data["open"]) /  df_sql_data["open"]) * 100

df_sql_data['WIN']  = np.where(df_sql_data['win5'] > 4.2, 'buy', '')

print(df_sql_data)
df_sql_data.to_excel('stock_h.xlsx', index=True)


print("Dataframe from SQL")

# Graficar los datos con la volatilidad y el volumen
''' 
mpf.plot(df_sql_data, type='candle', title='Price Graph with Volume and Volatility', ylabel='Price',
         addplot=[mpf.make_addplot([volatility] * len(df_sql_data), panel=1, color='orange', linestyle='--', ylabel='Volatility')],
         volume=True)
'''
# -----------------------

input_columns = ['d_high', 'd_low', 'd_close', 'volume', 'd_high5', 'd_low5', 'd_close5', 'd_high10', 'd_low10', 'd_close10', 'd_high30', 'd_low30', 'd_close30']

df_sql_data = df_sql_data.dropna()





x = df_sql_data[input_columns].values

y = df_sql_data['WIN'].values 
print("------- X ------ ")
print(x)




split_index = int(len(x) * 0.6)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Split the data into train and test sets
x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

y_train = np.where(y_train == 'buy', 1, 0)
y_test = np.where(y_test == 'buy', 1, 0)
print("------- Y ------ ")
print(y_train)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_scaled, y_train, epochs=10000, batch_size=320)

results = model.evaluate(x_test_scaled, y_test)

loss = results[0]
accuracy = results[1]

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


cursor.close()
connection.close()