#!/usr/bin/python3

# This script will create the clusters and assign trips to them.
# The tables ride_clusters and cluster_data will be emptied and refilled

from pyspark.sql import *
from schemas import *
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
import subprocess

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

registerTable(sqlCtx, Table.TRIP_TIMES)
def make_clusters(sample_size, kmeans_factor, suffix=None, writeIntermediate=False):

    suffix = str(int(sample_size / kmeans_factor)) if suffix is None else suffix

    print('Starting process for ' + suffix)

    registerTable(sqlCtx, Table.RAW_DATA)


    df = spark.sql("SELECT pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, id FROM trips")


    clusters = KMeans.train(df.select("pickup_longitude", "pickup_latitude")
                              .rdd.sample(False, (sample_size*100.0) / 77000000.0)
                              .map(lambda row: Vectors.dense(row["pickup_longitude"], row["pickup_latitude"])),
                            int(sample_size / kmeans_factor), 1000)


    def predict(long, lat):
        return clusters.predict((long, lat))

    results = df.rdd.map(lambda r: Row(r['id'], predict(r['pickup_longitude'], r['pickup_latitude']),
                                                predict(r['dropoff_longitude'], r['dropoff_latitude'])))

    rideClusters = spark.createDataFrame(results, rideClusterSchema)

    if writeIntermediate:
        rideClusters.write.mode('overwrite').parquet(hadoopify("clusters/ride_clusters" + suffix))


    clusterRows = []
    for i in range(0, len(clusters.clusterCenters)):
        center = clusters.clusterCenters[i]
        clusterRows.append(Row(i, center[0].item(), center[1].item()))
    spark.createDataFrame(clusterRows, schemaForTable(Table.CLUSTER_DATA)).write.save(hadoopify("clusters/cluster_data" + suffix),
                                                  format="parquet", mode="overwrite")

    clusters = None
    df = None
    clusterRows = None

    print(suffix + ': Clustering done.')

    rideClusters.createOrReplaceTempView('ride_clusters')
    combined = spark.sql("SELECT ride_id, pickup_cid, dropoff_cid, pickup_timeslot_id, dropoff_timeslot_id "
                         "FROM trip_times INNER JOIN ride_clusters ON (trip_id = ride_id)")

    if writeIntermediate:
        combined.write.mode('overwrite').parquet(hadoopify('clusters/combined_data' + suffix))
    combined.createOrReplaceTempView('combined_data')
    print(suffix + ': Combining done.')
    rideClusters = None

    cids = spark.sql('SELECT DISTINCT pickup_cid FROM combined_data')
    tids = spark.sql('SELECT DISTINCT pickup_timeslot_id FROM combined_data')

    grouping = spark.sql("SELECT pickup_timeslot_id, pickup_cid, COUNT(*) AS cnt "
                         + "FROM combined_data "
                         + "GROUP BY 1, 2 "
                         + "ORDER BY 1 ASC")

    grouping = {(tid, cid): cnt for (tid, cid, cnt) in grouping.collect()}
    print(suffix + ': Grouping done.')

    def get_count(tid, cid):
        return grouping.get((tid, cid), 0)

    counts = []

    pairs = tids.crossJoin(cids)

    res = pairs.rdd.map(lambda r: (r['pickup_timeslot_id'], r['pickup_cid'],
                                   get_count(r['pickup_timeslot_id'], r['pickup_cid'])))

    resDF = spark.createDataFrame(res, demandSchema)
    resDF.write.mode('overwrite').parquet(hadoopify('clusters/demand' + suffix))
    print(suffix + ': Demand done.')



#make_clusters(50000, 500) # ~100 clusters
#make_clusters(50000, 50)  # ~1000 clusters
#make_clusters(50000, 100) # ~500 clusters
make_clusters(50000, 250) # ~200 clusters
#make_clusters(50000, 67)  # ~750 clusters