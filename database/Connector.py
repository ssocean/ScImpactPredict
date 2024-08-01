import mysql.connector

# Connect to the MySQL database
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1q2w3e4r5t',
    database='literaturedatabase'
)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()
cursor.execute(r"SELECT * FROM literaturedatabase.literature;")
result = cursor.fetchall()  # Fetch all rows
# Python list
# my_list = ['item1', 'item2', 'item3']
#
# # Insert each element into the table
# for item in my_list:
#     query = "INSERT INTO ListElements (element) VALUES (%s)"
#     values = (item,)
#     cursor.execute(query, values)

# Commit the changes and close the connection
connection.commit()
cursor.close()
connection.close()