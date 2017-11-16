#!/usr/bin/python3

# THIS CODE IS USING 100 DAYS OF TRAINING DATA AND PRINTING A RMSE FOR THE FOLLOWING WEEK
# IT RETURNS RMSE OF PREDICTION FOR 10 CLUSTERS


from pyspark.sql import *
from demand_cache import invDemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

N_OF_CLUSTERS = 10   # number of clusters included
N_OF_TIME_SLOTS = 14400  # number of time slots that are being used for training
TIME_SLOTS_WITHIN_DAY = 144     # day is divided into that number of slots
FIRST_DAY_DAY_OF_WEEK = 3   # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

invDemandCache.init(spark, sqlCtx)

assembler = VectorAssembler(inputCols=["day_of_week", "time_of_day"], outputCol="features")
lr = LinearRegression(labelCol='demand')


def get_data(start_time, end_time, cluster):
    rows = []
    for tid in range(start_time, end_time):
        demand = invDemandCache.get_demand(tid, cluster)
        day_of_week = (FIRST_DAY_DAY_OF_WEEK + int(tid//144)) % 7
        time_of_day = tid % 144
        rows.append((day_of_week, time_of_day, demand))
        if tid % 100 == 0:
            print(tid)
    df = spark.createDataFrame(rows, ["day_of_week", "time_of_day", "demand"])
    output = assembler.transform(df)
    return output.select('features', 'demand')


errors = []
for cid in range(0, N_OF_CLUSTERS):
    train_data = get_data(0, N_OF_TIME_SLOTS, cid)
    test_data = get_data(N_OF_TIME_SLOTS, N_OF_TIME_SLOTS + 7 * TIME_SLOTS_WITHIN_DAY, cid)

    lr_model = lr.fit(train_data)
    test_results = lr_model.evaluate(test_data)
    error = test_results.rootMeanSquaredError
    errors.append((cid, error))


f = open("linear_regression_rmse.txt", "w")
f.write("Predictions are based on " + str(N_OF_TIME_SLOTS / TIME_SLOTS_WITHIN_DAY) + " days of data.\n\n")
for er in errors:
    f.write("RMSE on test data for cluster " + str(er[0]) + " on a week basis is " + str(er[1]) + "\n")
f.close()
