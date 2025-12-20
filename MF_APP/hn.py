import mysql.connector

conn = mysql.connector.connect(
    host="mutual-funds-price-prediction-model.cc1gsccemwsx.us-east-1.rds.amazonaws.com",
    user="admin",
    password="databaseconnector"
)

cursor = conn.cursor()
cursor.execute("CREATE DATABASE mydb")
conn.close()

