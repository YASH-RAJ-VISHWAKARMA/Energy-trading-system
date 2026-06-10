import sqlite3
from config import DATABASE_NAME

conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
cursor = conn.cursor()


def create_tables():
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        temperature REAL,
        demand REAL,
        current_price REAL,
        predicted_price REAL,
        decision TEXT
    )
    ''')

    conn.commit()


def insert_trade(data):
    cursor.execute('''
    INSERT INTO trades
    (timestamp, temperature, demand, current_price, predicted_price, decision)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', data)
    
    conn.commit()


def get_history():
    cursor.execute("SELECT * FROM trades ORDER BY id DESC")
    return cursor.fetchall()