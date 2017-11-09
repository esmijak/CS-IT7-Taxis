from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException
from enum import Enum, auto

combinedDataSchema = StructType([
  StructField("ride_id", IntegerType(), False),
  StructField("pickup_cid", IntegerType(), True),
  StructField("dropoff_cid", IntegerType(), True),
  StructField("pickup_timeslot_id", IntegerType(), True),
  StructField("dropoff_timeslot_id", IntegerType(), True)])

clusterDataSchema = StructType([
  StructField("ride_id", IntegerType(), False),
  StructField("cetroid_long", DoubleType(), True),
  StructField("centroid_lat", DoubleType(), True)])

timeSlotSchema = StructType([
  StructField("from", TimestampType(), False),
  StructField("to", TimestampType(), False),
  StructField("id", IntegerType(), False)])

rawDataSchema = StructType([
  StructField("tpep_pickup_datetime", TimestampType(), True),
  StructField("tpep_dropoff_datetime", TimestampType(), True),
  StructField("pickup_longitude", DoubleType(), True),
  StructField("pickup_latitude", DoubleType(), True),
  StructField("dropoff_longitude", DoubleType(), True),
  StructField("dropoff_latitude", DoubleType(), True),
  StructField("id", IntegerType(), False)])

demandSchema = StructType([
  StructField("pickup_timeslot_id", IntegerType(), False),
  StructField("pickup_cid", IntegerType(), False),
  StructField("demand", IntegerType(), True)])

rideClusterSchema = StructType([
  StructField("ride_id", IntegerType(), False),
  StructField("pickup_cid", IntegerType(), False),
  StructField("dropoff_cid", IntegerType(), False)])

tripTimeSchema = StructType([
  StructField("trip_id", IntegerType(), False),
  StructField("pickup_timeslot_id", IntegerType(), False),
  StructField("dropoff_timeslot_id", IntegerType(), False)
])

class Table(Enum):
  COMBINED_DATA = auto()
  CLUSTER_DATA = auto()
  TIME_SLOTS = auto()
  RAW_DATA = auto()
  DEMAND = auto()
  RIDE_CLUSTERS = auto()
  TRIP_TIMES = auto()

  COMBINED_DATA_SAMPLE = auto()
  CLUSTER_DATA_SAMPLE = auto()
  RAW_DATA_SAMPLE = auto()
  DEMAND_SAMPLE = auto()
  RIDE_CLUSTERS_SAMPLE = auto()
  TRIP_TIMES_SAMPLE = auto()

def tableName(tab):
  if tab is Table.COMBINED_DATA:
    return "combined_data"
  elif tab is Table.CLUSTER_DATA:
    return "cluster_data"
  elif tab is Table.TIME_SLOTS:
    return "time_slots"
  elif tab is Table.RAW_DATA:
    return "trips"
  elif tab is Table.DEMAND:
    return "demand"
  elif tab is Table.RIDE_CLUSTERS:
    return "ride_clusters"
  elif tab is Table.TRIP_TIMES:
    return "trip_times"
  elif tab is Table.COMBINED_DATA_SAMPLE:
    return "combined_data_sample"
  elif tab is Table.CLUSTER_DATA_SAMPLE:
    return "cluster_data_sample"
  elif tab is Table.RAW_DATA_SAMPLE:
    return "trips_sample"
  elif tab is Table.DEMAND_SAMPLE:
    return "demand_sample"
  elif tab is Table.RIDE_CLUSTERS_SAMPLE:
    return "ride_clusters_sample"
  elif tab is Table.TRIP_TIMES_SAMPLE:
      return "trip_times_sample"
  else:
    return None

def schemaForTable(tab):
  if tab is Table.COMBINED_DATA or tab is Table.COMBINED_DATA_SAMPLE:
    return combinedDataSchema
  elif tab is Table.CLUSTER_DATA or tab is Table.CLUSTER_DATA_SAMPLE:
    return clusterDataSchema
  elif tab is Table.TIME_SLOTS:
    return timeSlotSchema
  elif tab is Table.TRIP_TIMES or tab is Table.TRIP_TIMES_SAMPLE:
    return tripTimeSchema
  elif tab is Table.RAW_DATA or tab is Table.RAW_DATA_SAMPLE:
    return rawDataSchema
  elif tab is Table.DEMAND or tab is Table.DEMAND_SAMPLE:
    return demandSchema
  elif tab is Table.RIDE_CLUSTERS or tab is Table.RIDE_CLUSTERS_SAMPLE:
    return rideClusterSchema
  else:
    return None

def paramsForTable(tab):
  return (tableName(tab), schemaForTable(tab))

def loadDataFrame(sqlCtx, tab):
  (table, schema) = paramsForTable(tab)
  try:
    return sqlCtx.read.csv('hdfs://csit7-master:54310/user/csit7/%s/part-m-00000' % table, schema)
  except AnalysisException:
    return sqlCtx.read.parquet('hdfs://csit7-master:54310/user/csit7/%s' % table)

def registerTable(sqlCtx, tab):
  loadDataFrame(sqlCtx, tab).registerTempTable(tableName(tab))
