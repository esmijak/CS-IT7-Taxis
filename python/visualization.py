from schemas import *
from pyspark.sql import *
import matplotlib.pyplot as plt

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)


def monthly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql('SELECT month, COUNT(month) AS count '
                     'FROM (SELECT month(tpep_pickup_datetime) AS month FROM ' + tableName(Table.RAW_DATA) + ') '
                     'GROUP BY month '
                     'ORDER BY month').sample(False, 0.01).collect()

    months = list(map((lambda d: d['month']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(months, counts)
    plt.show()


def weekly_demand():
    registerTable(sqlCtx, Table.RAW_DATA)

    data = spark.sql("SELECT week, COUNT(week) AS count "
                     "FROM (SELECT weekofyear(tpep_pickup_datetime) AS week FROM " + tableName(Table.RAW_DATA) + ") "
                     "GROUP BY week "
                     "ORDER BY week").sample(False, 0.01).collect()

    weeks = list(map((lambda d: d['week']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(weeks, counts)
    plt.show()


def hourly_demand():
    registerTable(sqlCtx, Table.RAW_DATA_SAMPLE)

    data = spark.sql('SELECT hour, COUNT(hour) AS count '
                     'FROM (SELECT hour(tpep_pickup_datetime) AS hour FROM trips_sample) '
                     'GROUP BY hour '
                     'ORDER BY hour').collect()

    hours = list(map((lambda d: d['hour']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(hours, counts)
    plt.show()


def weekday_demand():
    registerTable(sqlCtx, Table.RAW_DATA_SAMPLE)

    data = spark.sql("SELECT weekday, COUNT(weekday) AS count "
                     "FROM (SELECT date_format(tpep_pickup_datetime, 'EEEE') AS weekday FROM trips_sample) "
                     "GROUP BY weekday "
                     "ORDER BY weekday").collect()

    weekdays = list(map((lambda d: d['weekday']), data))
    counts = list(map((lambda d: d['count']), data))

    plt.bar(weekdays, counts)
    plt.show()


weekly_demand()
