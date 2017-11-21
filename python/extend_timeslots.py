from pyspark.sql import *
from schemas import *

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

registerTable(sqlCtx, Table.TIME_SLOTS)
extended_timeslots = spark.sql('SELECT id, from, to, floor(3 + id / 144) % 7 AS day_of_week, '
                               'id % 144 AS time_of_day FROM time_slots')

extended_timeslots.write.mode('overwrite').parquet(hadoopify('extended_timeslots'))

