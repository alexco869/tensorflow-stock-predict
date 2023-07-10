import yfinance as yf
import mysql.connector


USERNAME="stocks"
PASSWORD="stocks"

df = yf.download("AAPL", start="2021-01-01", end="2021-07-01")
print(df)




connection = mysql.connector.connect(host='localhost',
                                         database='stocks',
                                         user=USERNAME,
                                         password=PASSWORD)

cursor = connection.cursor()

for _, row in df.iterrows():
    insert_query = "INSERT INTO historical_data (Date, open, high, low, close, adjClose, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    values = (row.name, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
    cursor.execute(insert_query, values)
    connection.commit()

cursor.close()
connection.close()
