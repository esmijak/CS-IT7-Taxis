#!/usr/bin/python3

# This script will create the clusters and assign trips to them.
# The tables ride_clusters and cluster_data will be emptied and refilled

from pyspark.sql import *
from schemas import *
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
import subprocess

# This constant determines how many random samples will be used for the clustering
# This only affects the creation of the clusters; ALL rides will be assigned to a cluster.
# Note that this amount of rows will be loaded into memory at once, and the clustering
# will require a large amount of RAM if this number is very large.
CLUSTERING_SIZE = 100000

# This constant is the average amount of pickups per cluster.
# That is, there will be CLUSTERING_SIZE / K_MEANS_FACTOR clusters.
K_MEANS_FACTOR = 250

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

# First we fetch the data we will use for clustering
print("Fetching {} random trips for initial clustering".format(CLUSTERING_SIZE))

registerTable(sqlCtx, Table.RAW_DATA)


df = spark.sql("SELECT pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, id FROM trips")


print("Done, now starting K-means with K={}".format(int(CLUSTERING_SIZE / K_MEANS_FACTOR)))

# Now we compute the clusters. This might take a while.

#km = KMeans(int(CLUSTERING_SIZE / K_MEANS_FACTOR))
clusters = KMeans.train(df.select("pickup_longitude", "pickup_latitude")
                          .rdd.sample(False, (CLUSTERING_SIZE*100.0) / 77000000.0)
                          .map(lambda row: Vectors.dense(row["pickup_longitude"], row["pickup_latitude"])),
                        int(CLUSTERING_SIZE / K_MEANS_FACTOR), 1000)


print("K-means is done, clearing any existing data...")

# Clean the database before proceeding

subprocess.call(["hadoop", "fs", "-rm", "-r", "-f", "/user/csit7/clusters/ride_clusters400"])
subprocess.call(["hadoop", "fs", "-rm", "-r", "-f", "/user/csit7/clusters/cluster_data400"])

print("Done, initiating refill")

# Now we first refill cluster_data, since we will need its ids later
#for c in zip(set(kmData.tolist()), centroids):
#  cur.execute("INSERT INTO taxi.cluster_data(cluster_id, centroid_long, centroid_lat) VALUES (%s, %s, %s)",
#               (int(c[0]), float(c[1][0]), float(c[1][1])))

def predict(long, lat):
    return clusters.predict((long, lat))

results = df.rdd.map(lambda r: Row(r['id'], predict(r['pickup_longitude'], r['pickup_latitude']),
                                            predict(r['dropoff_longitude'], r['dropoff_latitude'])))

sqlCtx.createDataFrame(results, schemaForTable(Table.RIDE_CLUSTERS))\
    .write.save(hadoopify("clusters/ride_clusters400"), format="parquet", mode="overwrite")

clusterRows = []
for i in range(0, len(clusters.clusterCenters)):
    center = clusters.clusterCenters[i]
    clusterRows.append(Row(i, center[0].item(), center[1].item()))
spark.createDataFrame(clusterRows, schemaForTable(Table.CLUSTER_DATA)).write.save(hadoopify("clusters/cluster_data400"),
                                              format="parquet", mode="overwrite")

print("All done, cleaning up and exiting")
