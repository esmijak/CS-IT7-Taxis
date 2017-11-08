from mpl_toolkits.basemap import Basemap

from schemas import *
from pyspark.sql import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)


def monthly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT month, COUNT(month) AS count '
                     'FROM (SELECT month(tpep_pickup_datetime) AS month FROM trips) '
                     'GROUP BY month '
                     'ORDER BY month').collect()

    months = list(map((lambda d: d['month']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(months, counts)
    plt.show()


def weekly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql("SELECT week, COUNT(week) AS count "
                     "FROM (SELECT weekofyear(tpep_pickup_datetime) AS week FROM trips) "
                     "GROUP BY week "
                     "ORDER BY week").collect()

    weeks = list(map((lambda d: d['week']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(weeks, counts)
    plt.show()


def hourly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT hour, COUNT(hour) AS count '
                     'FROM (SELECT hour(tpep_pickup_datetime) AS hour FROM trips) '
                     'GROUP BY hour '
                     'ORDER BY hour').collect()

    hours = list(map((lambda d: d['hour']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(hours, counts)
    plt.show()


def weekday_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql("SELECT weekday, COUNT(weekday) AS count "
                     "FROM (SELECT date_format(tpep_pickup_datetime, 'EEEE') AS weekday FROM trips) "
                     "GROUP BY weekday "
                     "ORDER BY weekday").collect()

    weekdays = list(map((lambda d: d['weekday']), data))
    counts = list(map((lambda d: d['count']), data))

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
                     resolution='h', area_thresh=1,
                     llcrnrlon=-74.20, llcrnrlat=40.55,
                     urcrnrlon=-73.68, urcrnrlat=40.95)

    my_map.drawcountries()
    my_map.fillcontinents(color='coral', lake_color='aqua', zorder=0)
    my_map.drawcoastlines()
    my_map.drawmapboundary(fill_color='aqua')

    my_map.readshapefile('./geo_export_84b991a0-49f4-4c35-a58a-f770d4bb88fc', '1', zorder=1, color='gray')

    x, y = my_map(long, lat)
    my_map.plot(x, y, 'bo', markersize=5, alpha=0.3)

    plt.show()


def centroid_demand(normalization=0):
    registerTable(sqlCtx, Table.CLUSTER_DATA)
    registerTable(sqlCtx, Table.RIDE_CLUSTERS)

    data = spark.sql("SELECT cetroid_long, centroid_lat, count "
                     "FROM "
                     "(SELECT pickup_cid, COUNT(*) as count "
                     "FROM ride_clusters GROUP BY pickup_cid) AS rides "
                     "INNER JOIN "
                     "cluster_data ON rides.pickup_cid = cluster_data.ride_id "
                     ).collect()

    centroidsX = list(map(lambda row: row['cetroid_long'], data))
    centroidsY = list(map(lambda row: row['centroid_lat'], data))
    count = list(map(lambda row: row['count'], data))
    if normalization != 0:
        count = list(map(lambda c: normalization if c > normalization  else c, count))

    my_map = Basemap(projection='merc', lat_0=40.78, lon_0=-73.96,
                     resolution='f', area_thresh=1,
                     llcrnrlon=-74.20, llcrnrlat=40.55,
                     urcrnrlon=-73.68, urcrnrlat=40.95)

    my_map.drawcountries()
    my_map.fillcontinents(color='coral', lake_color='aqua', zorder=0)
    my_map.drawcoastlines()
    my_map.drawmapboundary(fill_color='aqua')

    my_map.readshapefile('./geo_export_84b991a0-49f4-4c35-a58a-f770d4bb88fc', '1', zorder=1)

    x, y = my_map(centroidsX, centroidsY)
    my_map.scatter(x, y, c=count, marker="o", cmap=cm.summer, alpha=1, zorder=2)
    my_map.colorbar()

    plt.show()


#call visualizations here
hourly_demand()

