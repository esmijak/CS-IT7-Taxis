#import psycopg2
#from psycopg2.extras import execute_batch

from pyspark.sql import *
from schemas import *
from pyspark.storagelevel import StorageLevel

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

START_INDEX = 0
END_INDEX = 100000

print('Fetching grouping')
grouping = spark.read.parquet(hadoopify('temp_grouping'))
grouping = {(tid, cid): cnt for (tid, cid, cnt) in grouping.collect()}

def get_count(tid, cid):
    return grouping.get((tid, cid), 0)

print("Fetching pairs {} - {} ({} in total)".format(START_INDEX, END_INDEX, END_INDEX - START_INDEX))
pairs = spark.read.parquet(hadoopify('timeslot_cluster_combos')).rdd\
                  .zipWithIndex()\
                  .filter(lambda z: z[1] >= START_INDEX and z[1] < END_INDEX)

res = pairs.map(lambda r: (r['pickup_timeslot_id'], r['pickup_cid'],
                               get_count(r['pickup_timeslot_id'], r['pickup_cid'])))

print("Inserting")

resDF = spark.createDataFrame(res, demandSchema)
resDF.persist(StorageLevel.DISK_ONLY)
resDF.write.mode('overwrite').parquet(hadoopify('demand'))
print("Done")
