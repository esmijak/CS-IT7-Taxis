#import psycopg2
#from psycopg2.extras import execute_batch

from pyspark.sql import *
from schemas import *

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

print("Fetching raw values")
registerTable(sqlCtx, Table.COMBINED_DATA)

cids = spark.sql('SELECT DISTINCT pickup_cid FROM combined_data')
tids = spark.sql('SELECT DISTINCT pickup_timeslot_id FROM combined_data')

print("Creating grouping")

grouping = spark.sql("SELECT pickup_timeslot_id, pickup_cid, COUNT(*) AS cnt "
            + "FROM combined_data "
            + "GROUP BY 1, 2")

grouping = {(tid, cid): cnt for (tid, cid, cnt) in grouping.collect()}

def get_count(tid, cid):
    return grouping.get((tid, cid), 0)

counts = []

print("Formatting data")
pairs = tids.crossJoin(cids)

res = pairs.rdd.map(lambda r: (r['pickup_timeslot_id'], r['pickup_cid'],
                               get_count(r['pickup_timeslot_id'], r['pickup_cid'])))

print("Inserting")
resDF = spark.createDataFrame(res, demandSchema)
resDF.write.mode('overwrite').parquet('/users/csit7/demand')
print("Done")
