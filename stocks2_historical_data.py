import pandas as pd
import mysql.connector
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import mplfinance as mpf

today = date.today()
USERNAME="stocks"
PASSWORD="stocks"
connection = mysql.connector.connect(host='localhost',
                                    database='stocks',
                                    user=USERNAME,
                                    password=PASSWORD)

cursor = connection.cursor()

cursor.execute("SELECT symbol FROM stocks2")

myresult = cursor.fetchall()

elapsed = relativedelta(years=10)

i = 0
for r in myresult:
    df = yf.download(r, start=today-elapsed, end=today)
    name=r[0]
    for _, row in df.iterrows():
        insert_query = "INSERT INTO historical_data (Date, open, high, low, close, adjClose, volume, symbol) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        values = (row.name, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'], name)
        cursor.execute(insert_query, values)
        connection.commit()

cursor.close()
connection.close()