from  stock_ai import stock_ai
from datetime import datetime
import yfinance as yf
import mysql.connector
import pandas as pd
from datetime import date

USERNAME="stocks"
PASSWORD="stocks"

connection = mysql.connector.connect(host='localhost',
                                         database='stocks',
                                         user=USERNAME,
                                         password=PASSWORD)

cursor = connection.cursor()

sql_query = "SELECT * FROM historical_data"

cursor.execute(sql_query)

df_sql_data = pd.DataFrame.from_records(cursor.fetchall(), columns=[x[0] for x in cursor.description])

for _, row in df_sql_data.iterrows():
    stock = stock_ai(row['symbol'], datetime(2023, 7, 12), 3650)
    stock.update()


obj = stock_ai('V', datetime(2023, 7, 12), 3650)

obj.update()

obj.fill_deltas()

obj.train(3000)

obj.test()

obj.evaluate