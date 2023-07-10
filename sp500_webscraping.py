import pandas as pd
import mysql.connector
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]

connection = mysql.connector.connect(host='localhost',
                                    database='stocks',
                                    user='alex',
                                    password='alex2000')
df = pd.DataFrame(sp500)
df = df.dropna()

cursor = connection.cursor()
for _, row in df.iterrows():
    insert_query = "INSERT INTO stocks2 (Symbol, Security, Sector, Sub_Industry, Location, Date, CIK, Founded) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    values = tuple(row)
    cursor.execute(insert_query, values)
    connection.commit()
