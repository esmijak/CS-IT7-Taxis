from schemas import *
from pyspark.sql import *
#from feature_cache import demandCache
from fi_features_cache import gridCache
import math
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

"""Cluster related constant: """
N_OF_LAT = 2   # number of clusters used : all of them
"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
WEEK_NB_TEST = 23 # start of june
FIRST_WEEK = 1
LAST_WEEK = 27
DAY_IN_WEEK = 7


spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
registerTable(sqlCtx, Table.FINAL_FEATURES)

gridTable = spark.read.parquet(hadoopify('grids/final_features_grid'))


gridCache.init(spark, sqlCtx)

def getFeatures(week_nb, day_of_week, time_of_day_code) :
    df = sqlCtx.sql('SELECT * FROM final_features WHERE week = {} AND day_of_week = {} AND time_of_day_code = {}'.format(week_nb, day_of_week, time_of_day_code))
    col = df.first()
    day = col['day']
    hour = col['hour']
    minute = col['minute']
    return day, hour, minute

def getGridInfo(long, lat) :
    isManhattan = 0
    isAirport = 0
    if long > -74.025 and long < -73.975 and lat > 40.705 and lat < 40.75 :
        isManhattan = 1
    elif long > -73.81 and long < -73.77 and lat > 40.64 and lat < 40.663 :
        isAirport = 1
    return isManhattan, isAirport



def extract_feature(curFeature) :
    week = curFeature['week']
    day = curFeature['day']
    time_of_day_code = curFeature['time_of_day_code']
    day_of_week = curFeature['day_of_week']
    hour = curFeature['hour']
    minute = curFeature['minute']
    is_manhattan = curFeature['is_manhattan']
    is_airport = curFeature['is_airport']
    amount = curFeature['amount']
    return time_of_day_code, day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount

def main():
    rows = []
    for week_nb in range(FIRST_WEEK, LAST_WEEK + 1):
        print('week nb : ', week_nb)
        for day_of_week in range(DAY_IN_WEEK):
            for time_of_day_code in range(TIME_SLOTS_WITHIN_DAY):
                for long in range(N_OF_LAT):
                    for lat in range (N_OF_LAT):
                        features = gridCache.get_demand_grid(week_nb, day_of_week, time_of_day_code, long, lat)

                        if len(features) > 0:
                            curFeature = features
                            time_of_day_code, day_of_week, day, week, hour, minute, is_manhattan, is_airport, amount = extract_feature(curFeature)
                        else:
                            day, hour, minute = getFeatures(week_nb, day_of_week, time_of_day_code)
                            is_manhattan, is_airport = getGridInfo(long, lat)
                            amount = 0


                        rows.append((time_of_day_code, day_of_week, day, week_nb, hour, minute, lat, long, is_manhattan, is_airport, amount))


    df = spark.createDataFrame(rows,
                               ["time_of_day_code", "day_of_week", "day", "week", "hour", "minute", "pickup_lat_slot", "pickup_long_slot","is_manhattan", "is_airport",
                                "amount"])

    df.write.mode('overwrite').parquet(hadoopify('grid/final_features_grid_0'))

#import cProfile

#cProfile.run('main()')
main()