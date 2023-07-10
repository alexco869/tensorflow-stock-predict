import mysql.connector



USERNAME="stocks"
PASSWORD="stocks"
try:
    connection = mysql.connector.connect(host='localhost',
                                         database='stocks',
                                         user= USERNAME,
                                         password=PASSWORD)

    mySql_insert_query = """INSERT INTO stocks (symbol, name, shares) 
                           VALUES 
                           ('AAPL', 'Apple Inc.', 15787154000) """

    cursor = connection.cursor()
    cursor.execute(mySql_insert_query)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into stocks table")
    cursor.close()

except mysql.connector.Error as error:
    print("Failed to insert record into stocks table {}".format(error))

finally:
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

