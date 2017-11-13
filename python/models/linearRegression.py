#!/usr/bin/python3

# THIS CODE IS USING 100 DAYS OF TRAINING DATA AND PRINTING A RMSE FOR THE FOLLOWING DAY
# IT RETURNS RMSE OF PREDICTION FOR 10 CLUSTERS


from pyspark.sql import *
from demand_cache import invDemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

N_OF_CLUSTERS = 10   # number of clusters included
N_OF_TIME_SLOTS = 14400  # number of time slots that are being used for training
TIME_SLOTS_WITHIN_DAY = 144     # day is divided into that number of slots


spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

invDemandCache.init(spark, sqlCtx)

assembler = VectorAssembler(inputCols=["pickup_timeslot_id"], outputCol="features")
lr = LinearRegression(labelCol='demand')


errors = []
for cid in range(0, N_OF_CLUSTERS):
    rows = []
    for tid in range(0, N_OF_TIME_SLOTS):
        demand = invDemandCache.get_demand(tid, cid)
        rows.append((tid, demand))
        if tid % 100 == 0:
            print(tid)
    df = spark.createDataFrame(rows, ["pickup_timeslot_id", "demand"])
    output = assembler.transform(df)
    train_data = output.select('features', 'demand')

    rows = []
    for tid in range(N_OF_TIME_SLOTS, N_OF_TIME_SLOTS + 7 * TIME_SLOTS_WITHIN_DAY):
        demand = invDemandCache.get_demand(tid, cid)
        rows.append((tid, demand))
    df = spark.createDataFrame(rows, ["pickup_timeslot_id", "demand"])
    output = assembler.transform(df)
    test_data = output.select('features', 'demand')

    lr_model = lr.fit(train_data)
    test_results = lr_model.evaluate(test_data)
    error = test_results.rootMeanSquaredError
    errors.append((cid, error))


f = open("linear_regression_rmse.txt", "w")
f.write("Predictions are based on " + str(N_OF_TIME_SLOTS / TIME_SLOTS_WITHIN_DAY) + " days of data.\n\n")
for er in errors:
    f.write("RMSE on test data for cluster " + str(er[0]) + " on a week basis is " + str(er[1]) + "\n")
f.close()
