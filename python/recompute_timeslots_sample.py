#!/usr/bin/python3

import psycopg2
import datetime

start_slot = datetime.datetime(2015, 1, 1, 0)
slot_list = []
while (start_slot < datetime.datetime(2016, 1, 1, 0)):
    next = start_slot + datetime.timedelta(minutes=10)
    print("Now at {}.".format(next.isoformat()))
    slot_list.append(tuple((start_slot, next)))
    start_slot = next

sql = "INSERT INTO taxi.time_slots_sample VALUES(%s, %s)"
conn = None
try:
    conn = psycopg2.connect("dbname=taxidata user=csit7")
    cur = conn.cursor()
    cur.executemany(sql, slot_list)
    conn.commit()
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
