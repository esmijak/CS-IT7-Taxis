from pyspark.sql.types import *

combinedDataSchema = StructType([StructField("ride_id", IntegerType(), False), StructField("pickup_cid", IntegerType(), True), StructField("dropoff_cid", IntegerType(), True), StructField("pickup_timeslot_id", IntegerType(), True), StructField("dropoff_timeslot_id", IntegerType(), True)])

clusterDataSchema = StructType([StructField("ride_id", IntegerType(), False), StructField("cetroid_long", DoubleType(), True), StructField("centroid_lat", DoubleType(), True)])

timeSlotSchema = StructType([StructField("from", TimestampType(), False), StructField("to", TimestampType(), False), StructField("id", IntegerType(), False)])
