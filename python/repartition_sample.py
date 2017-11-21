#!/usr/bin/python3

from pyspark.sql import *
from pyspark.sql.types import *
from schemas import *
import datetime

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

print("Fetching data...")

tripsDF = loadDataFrame(sqlCtx, Table.RAW_DATA_SAMPLE)
slots = loadDataFrame(sqlCtx, Table.TIME_SLOTS).collect()
slot_count = len(slots)

print("Done ({} slots for {} trips)".format(slot_count, tripsDF.count()))

time_per_slot = (60 * 60 * 24 * 180) / slot_count
start_time = datetime.datetime(2015, 1, 1, 0).timestamp()
def find_slot(time):
    return int((time.timestamp() - start_time) / time_per_slot)

def handle_trip(agg, row):
  pickup_time = row['tpep_pickup_datetime']
  dropoff_time = row['tpep_dropoff_datetime']
  pickup_slot = find_slot(pickup_time)
  dropoff_slot = find_slot(dropoff_time)
  data = (row['id'], pickup_slot, dropoff_slot)
  agg.append(data)
  return agg

print("Starting aggregation")
res = tripsDF.rdd.aggregate([], handle_trip, lambda x, y: x + y)
print("Aggregation done, writing results")

resDF = sqlCtx.createDataFrame(res, tripTimeSchema)
resDF.write.mode("overwrite").parquet(hadoopify("trip_times_sample"))
print("Done")