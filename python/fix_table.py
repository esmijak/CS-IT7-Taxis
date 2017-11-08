from pyspark.sql import *
import subprocess

####
TO_FIX = "trips"
####

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

def remove_existing(name):
    subprocess.call(["hadoop", "fs", "-rm", "-r", "-f", "/user/csit7/" + name])

def transfer(name):
    df = sqlCtx.read.jdbc(url="jdbc:postgresql://csit7-master/taxidata", table="taxi."+name,
                          properties={"user":"csit7", "password": "taxi2017"})
    remove_existing(name)
    df.write.mode("append").format("parquet").partitionBy("id")\
        .save("hdfs://csit7-master:54310/user/csit7/" + name)

transfer(TO_FIX)