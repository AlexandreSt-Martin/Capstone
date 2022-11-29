from flask import Flask, render_template, Response, flash
import os
from flask import request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from collections import deque

from sqlalchemy.sql import func

app = Flask(__name__)

#set name of database for code
db_name = 'Database.db'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

#initialize database, used for SQLAlchemy commands
db = SQLAlchemy(app)

class Plate(db.Model):
    __tablename__ = 'LicensePlates'
    LicensePlate = db.Column(db.String, primary_key=True, nullable=False, unique=True)
    Owner = db.Column(db.String, nullable=False)
    Info = db.Column(db.String, nullable=False)

class Criminal(db.Model):
    __tablename__ = 'Criminals'
    Name = db.Column(db.String, primary_key=True, nullable=False, unique=True)
    Crime = db.Column(db.String, nullable=False)

#compare string detected from license plate to plate table in database
def plate_detected(str):
    conn = sqlite3.connect('Database.db')
    cur=conn.cursor()
    cur.execute("SELECT * FROM LicensePlate WHERE one=?", (columnchosen,))
     
    records = cur.fetchall()
    for row in records:
        if(row[0] == str):
            d = deque(row[0], row[1], row[2])
    cur.close()
    return d
    
#compare string of person's name to database
def compare_face(str):
    conn = sqlite3.connect('Database.db')
    cur=conn.cursor()
    cur.execute("SELECT * FROM Criminals WHERE one=?", (columnchosen,))
	
    records = curr.fetchall()
    for row in records:
        if(row[0] == str):
            d = deque(row[0], row[1])
    cur.close()
    return d

#add License Plate to database
def add_plate(LicensePlate, Owner, Info):
	try:
		con = sql.connect('Database.db')
		c = con.cursor()
		c.execute("INSERT INTO LicensePlate (LicensePlate, Owner, Info) VALUES (%s, %s, %s)" %(LicensePlate, Owner, Info))
		con.commit()
	except:
		print("Error adding plate to db")

#add Face to database
def add_face(Name, Crime):
	try:
		con=sql.connect('Database.db')
		c=con.cursor()
		c.execute("INSERT INTO Criminals (Name, Crime) VALUES (%b, %s, %s)" %(Name, Crime))
		con.commit()
	except:
		print("Error adding face to db")
