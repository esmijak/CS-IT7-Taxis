from mpl_toolkits.basemap import Basemap

from schemas import *
from pyspark.sql import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)


def monthly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT month, COUNT(month) AS cnt '
                     'FROM (SELECT month(tpep_pickup_datetime) AS month FROM trips) '
                     'GROUP BY month '
                     'ORDER BY month').collect()

    months = list(map((lambda d: d['month']), data))
    counts = list(map((lambda d: d['cnt']), data))

    fig, ax = plt.subplots()
    ax.set_xticklabels(('0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'))

    plt.bar(months, counts)
    plt.show()


def weekly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql("SELECT week, COUNT(week) AS cnt "
                     "FROM (SELECT weekofyear(tpep_pickup_datetime) AS week FROM trips) "
                     "GROUP BY week "
                     "ORDER BY week").collect()

    weeks = list(map((lambda d: d['week']), data))
    counts = list(map((lambda d: d['cnt']), data))

    plt.bar(weeks, counts)
    plt.show()


def hourly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT hour, COUNT(hour) AS cnt '
                     'FROM (SELECT hour(tpep_pickup_datetime) AS hour FROM trips) '
                     'GROUP BY hour '
                     'ORDER BY hour').collect()

    hours = list(map((lambda d: d['hour']), data))
    counts = list(map((lambda d: d['cnt']), data))

    plt.bar(hours, counts)
    plt.show()


def weekday_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql("SELECT weekday, COUNT(weekday) AS cnt "
                     "FROM (SELECT date_format(tpep_pickup_datetime, 'u') AS weekday FROM trips) "
                     "GROUP BY weekday").collect()

    weekdays = list(map((lambda d: d['weekday']), data))
    counts = list(map((lambda d: d['cnt']), data))

    fig, ax = plt.subplots()
    ax.set_xticklabels('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun')
    plt.bar(weekdays, counts)
    plt.show()


def points_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT pickup_longitude, pickup_latitude '
                     'FROM trips').sample(False, 0.005).collect()

    long = list(map(lambda row: row['pickup_longitude'], data))
    lat = list(map(lambda row: row['pickup_latitude'], data))
    long = list(filter(lambda l: abs(l) > 0, long))
    lat = list(filter(lambda l: abs(l) > 0, lat))

    my_map = Basemap(projection='merc', lat_0=40.78, lon_0=-73.96,
                     resolution='f', area_thresh=1,
                     llcrnrlon=-74.20, llcrnrlat=40.55,
                     urcrnrlon=-73.68, urcrnrlat=40.95)

    my_map.drawcountries()
    my_map.fillcontinents(color='#d4d4d4', lake_color='#a7cdf2', zorder=0)
    my_map.drawcoastlines()
    my_map.drawmapboundary(fill_color='#a7cdf2')

    my_map.readshapefile('./geo_export_84b991a0-49f4-4c35-a58a-f770d4bb88fc', '1', zorder=1, color='gray')

    x, y = my_map(long, lat)
    my_map.plot(x, y, 'bo', markersize=5, alpha=0.3)

    plt.show()


def centroid_demand():
    registerTable(sqlCtx, Table.CLUSTER_DATA)
    registerTable(sqlCtx, Table.RIDE_CLUSTERS)

    data = spark.sql("SELECT cetroid_long, centroid_lat, count "
                     "FROM "
                     "(SELECT pickup_cid, COUNT(*) as count "
                     "FROM ride_clusters GROUP BY pickup_cid) AS rides "
                     "INNER JOIN "
                     "cluster_data ON rides.pickup_cid = cluster_data.ride_id "
                     "WHERE rides.pickup_cid < 360 "
                     ).collect()

    centroidsX = list(map(lambda row: row['cetroid_long'], data))
    centroidsY = list(map(lambda row: row['centroid_lat'], data))
    count = list(map(lambda row: 2**log(row['count'], 10), data))

    my_map = Basemap(projection='merc', lat_0=40.78, lon_0=-73.96,
                     resolution='h', area_thresh=1,
                     llcrnrlon=-74.20, llcrnrlat=40.55,
                     urcrnrlon=-73.68, urcrnrlat=40.95)

    my_map.drawcountries()
    my_map.fillcontinents(color='#d4d4d4', lake_color='#a7cdf2', zorder=0)
    my_map.drawcoastlines()
    my_map.drawmapboundary(fill_color='#a7cdf2')

    my_map.readshapefile('./geo_export_84b991a0-49f4-4c35-a58a-f770d4bb88fc', '1', zorder=1)

    x, y = my_map(centroidsX, centroidsY)
    my_map.scatter(x, y, c=count, marker="o", cmap=cm.autumn_r, alpha=1, zorder=2)
    my_map.colorbar()

    plt.show()


def grid_demand():
    grid = spark.read.parquet(hadoopify('grids/final_features_grid'))
    grid.registerTempTable("grid")

    data = spark.sql("SELECT pickup_long_slot, pickup_lat_slot, COUNT(*) AS count "
                     "FROM grid "
                     "GROUP BY pickup_long_slot, pickup_lat_slot "
                     ).collect()
    east = -74.15
    west = -73.72
    south = 40.58
    north = 40.85

    horizontal_slots = 25
    vertical_slots = 25

    longitude_step = (west - east) / float(horizontal_slots)
    latitude_step = (north - south) / float(vertical_slots)

    gridX = list(map(lambda row: row['pickup_long_slot'], data))
    gridX = list(map(lambda x: x*longitude_step + east, gridX))
    gridY = list(map(lambda row: row['pickup_lat_slot'], data))
    gridY = list(map(lambda y: y*latitude_step + south, gridY))
    count = list(map(lambda row: 2**log(row['count'], 10), data))

    my_map = Basemap(projection='merc', lat_0=40.78, lon_0=-73.96,
                     resolution='f', area_thresh=1,
                     llcrnrlon=-74.20, llcrnrlat=40.55,
                     urcrnrlon=-73.68, urcrnrlat=40.95)

    my_map.drawcountries()
    my_map.fillcontinents(color='#d4d4d4', lake_color='#a7cdf2', zorder=0)
    my_map.drawcoastlines()
    my_map.drawmapboundary(fill_color='#a7cdf2')

    my_map.readshapefile('./geo_export_84b991a0-49f4-4c35-a58a-f770d4bb88fc', '1', zorder=1)

    x, y = my_map(gridX, gridY)
    my_map.scatter(x, y, c=count, marker="o", cmap=cm.autumn_r, alpha=1, zorder=2)
    my_map.colorbar()

    plt.show()


# call visualizations here
# https://data.cityofnewyork.us/api/geospatial/xr67-eavy?method=export&format=Shapefile
# For map visualization, download this file first (shapefile of NYC streets)

# monthly_demand()
#hourly_demand()
# weekday_demand()
weekly_demand()
# points_demand()
# grid_demand()
