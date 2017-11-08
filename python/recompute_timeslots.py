#!/usr/bin/python3

from pyspark.sql import *
from schemas import timeSlotSchema
import datetime

start_slot = datetime.datetime(2015, 1, 1, 0)
slot_list = []
id = 0
while (start_slot < datetime.datetime(2015, 6, 30, 0)):
    next = start_slot + datetime.timedelta(minutes=10)
    slot_list.append(tuple((start_slot, next, id)))
    id += 1
    start_slot = next

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
df = spark.createDataFrame(slot_list, timeSlotSchema)

df.write.mode("overwrite").parquet("/user/csit7/time_slots");
