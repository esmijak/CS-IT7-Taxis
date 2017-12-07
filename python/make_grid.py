from pyspark.sql import *
from schemas import *

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

suffix = '_ngrid2500'
missing_value = None # Value which will be stored for locations outside of the grid. Should be an integer or None

east = -74.15
west = -73.72
south = 40.58
north = 40.85

horizontal_slots = 25
vertical_slots = 25

longitude_step = (west - east) / float(horizontal_slots)
latitude_step = (north - south) / float(vertical_slots)

###
def calc_longitude_slot(long):
    norm = long - east
    slot = int(norm / longitude_step)
    return slot if slot >= 0 and slot < horizontal_slots else missing_value

def calc_latitude_slot(lat):
    norm = lat - south
    slot = int(norm / latitude_step)
    return slot if slot >= 0 and slot < vertical_slots else missing_value

def map_row(row):
    pickup_long = calc_longitude_slot(row['pickup_longitude'])
    pickup_lat = calc_latitude_slot(row['pickup_latitude'])
    dropoff_long = calc_longitude_slot(row['dropoff_longitude'])
    dropoff_lat = calc_latitude_slot(row['dropoff_latitude'])
    return Row(row['id'], pickup_long, pickup_lat, dropoff_long, dropoff_lat)

trips = loadDataFrame(sqlCtx, Table.RAW_DATA)
mapped = spark.createDataFrame(trips.rdd.map(map_row), gridTripSchema)

data = spark.createDataFrame([(east, west, south, north, horizontal_slots, vertical_slots, missing_value)], gridDataSchema)
data.write.mode('overwrite').parquet(hadoopify('grids/griddata' + suffix))
mapped.write.mode('overwrite').parquet(hadoopify('grids/tripdata' + suffix))

mapped.createOrReplaceTempView('mapped')
#spark.read.parquet('/user/csit7/grids/tripdata' + suffix).createOrReplaceTempView('mapped')

registerTable(sqlCtx, Table.TRIP_TIMES)
combined = spark.sql("""
SELECT mapped.trip_id AS trip_id, pickup_long_slot, pickup_lat_slot,
       dropoff_long_slot, dropoff_lat_slot, pickup_timeslot_id, dropoff_timeslot_id
FROM mapped INNER JOIN trip_times ON (mapped.trip_id = trip_times.trip_id)
WHERE     pickup_long_slot IS NOT NULL
      AND pickup_lat_slot IS NOT NULL
      AND dropoff_long_slot IS NOT NULL
      AND dropoff_lat_slot IS NOT NULL
""".format(vertical_slots, vertical_slots))
combined.write.mode('overwrite').parquet(hadoopify('grids/combined' + suffix))
combined.createOrReplaceTempView('combined')

demand = spark.sql("""
SELECT pickup_timeslot_id, pickup_long_slot, pickup_lat_slot, COUNT(*) AS cnt
FROM combined
GROUP BY 1, 2, 3""")
demand.write.mode('overwrite').parquet(hadoopify('grids/demand' + suffix))