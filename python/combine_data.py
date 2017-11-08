from pyspark.sql import *
from schemas import *

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

registerTable(sqlCtx, Table.TRIP_TIMES)
registerTable(sqlCtx, Table.RIDE_CLUSTERS)

combined = spark.sql("SELECT ride_id, pickup_cid, dropoff_cid, pickup_timeslot_id, dropoff_timeslot_id FROM trip_times INNER JOIN ride_clusters ON (trip_id = ride_id)")

combined.write.mode('overwrite').parquet('/user/csit7/combined_data')