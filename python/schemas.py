from pyspark.sql.types import *
from enum import Enum, auto

combinedDataSchema = StructType([StructField("ride_id", IntegerType(), False), StructField("pickup_cid", IntegerType(), True), StructField("dropoff_cid", IntegerType(), True), StructField("pickup_timeslot_id", IntegerType(), True), StructField("dropoff_timeslot_id", IntegerType(), True)])

clusterDataSchema = StructType([StructField("ride_id", IntegerType(), False), StructField("cetroid_long", DoubleType(), True), StructField("centroid_lat", DoubleType(), True)])

timeSlotSchema = StructType([StructField("from", TimestampType(), False), StructField("to", TimestampType(), False), StructField("id", IntegerType(), False)])

rawDataSchema = StructType([StructField("tpep_pickup_datetime", TimestampType(), True), StructField("tpep_dropoff_datetime", TimestampType(), True), StructField("pickup_longitude", DoubleType(), True), StructField("pickup_latitude", DoubleType(), True), StructField("dropoff_longitude", DoubleType(), True), StructField("dropoff_latitude", DoubleType(), True), StructField("id", IntegerType(), False)])

class Table(Enum):
  COMBINED_DATA = auto()
  CLUSTER_DATA = auto()
  TIME_SLOTS = auto()
  RAW_DATA = auto()

  COMBINED_DATA_SAMPLE = auto()
  CLUSTER_DATA_SAMPLE = auto()
  RAW_DATA_SAMPLE = auto()

def tableName(tab):
  if tab is Table.COMBINED_DATA:
    return "combined_data"
  elif tab is Table.CLUSTER_DATA:
    return "cluster_data"
  elif tab is Table.TIME_SLOTS:
    return "time_slots"
  elif tab is Table.RAW_DATA:
    return "trips"
  elif tab is Table.COMBINED_DATA_SAMPLE:
    return "combined_data_sample"
  elif tab is Table.CLUSTER_DATA_SAMPLE:
    return "cluster_data_sample"
  elif tab is Table.RAW_DATA_SAMPLE:
    return "trips_sample"
  else:
    return None

def schemaForTable(tab):
  if tab is Table.COMBINED_DATA or tab is Table.COMBINED_DATA_SAMPLE:
    return combinedDataSchema
  elif tab is Table.CLUSTER_DATA or tab is Table.CLUSTER_DATA_SAMPLE:
    return clusterDataSchema
  elif tab is Table.TIME_SLOTS:
    return timeSlotSchema
  elif tab is Table.RAW_DATA or tab is Table.RAW_DATA_SAMPLE:
    return rawDataSchema
  else:
    return None

def paramsForTable(tab):
  return (tableName(tab), schemaForTable(tab))

def loadDataFrame(sqlCtx, tab):
  (table, schema) = paramsForTable(tab)
  return sqlCtx.read.csv('hdfs://172.25.24.242:54310/user/csit7/%s/part-m-00000' % table, schema)

def registerTable(sqlCtx, tab):
  loadDataFrame(sqlCtx, tab).registerTempTable(tableName(tab))
