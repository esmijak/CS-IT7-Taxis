from pyspark.sql import *
from schemas import *

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

registerTable(sqlCtx, Table.TRIP_TIMES)
#registerTable(sqlCtx, Table.RIDE_CLUSTERS)

clusters = spark.read.parquet(hadoopify('clusters/ride_clusters400'))
clusters.createOrReplaceTempView('ride_clusters')


combined = spark.sql("SELECT ride_id, pickup_cid, dropoff_cid, pickup_timeslot_id, dropoff_timeslot_id "
                     "FROM trip_times INNER JOIN ride_clusters ON (trip_id = ride_id)")

combined.write.mode('overwrite').parquet(hadoopify('clusters/combined_data400'))