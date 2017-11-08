
import psycopg2
from sklearn.cluster import KMeans
from time import time
import numpy as np



conn = psycopg2.connect("dbname=taxidata user=csit7")
cur = conn.cursor()
cur.execute("SELECT pickup_longitude, pickup_latitude FROM taxi.trips_sample TABLESAMPLE SYSTEM (1) ")


all_data = cur.fetchall()

#random sampling
learning_data = all_data[]


learning_longitude = [d[0] for d in all_data]
learning_latitude = [d[1] for d in all_data]





conn.close()