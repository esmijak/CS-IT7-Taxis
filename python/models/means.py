#!/usr/bin/python3

from pyspark.sql import *
from demand_cache import demandCache

N_OF_CLUSTERS = 10   # number of clusters for which mean is being calculated
N_OF_TIME_SLOTS = 14400  # number of time slots that are being used for training
TIME_SLOTS_WITHIN_DAY = 144     # day is divided into that number of slots
THRESHOLD = 5   # difference between predicted and actual number of rides within
                # which the prediction is classified as correct

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

demandCache.init(spark, sqlCtx)

# initializing dir in which means are stored in format (cluster_id, time_of_day, weekday)
data = {}
for day_time in range(0, TIME_SLOTS_WITHIN_DAY):
    for weekday in range(0, 7):
        for cid in range(0, N_OF_CLUSTERS):
            data[(cid, day_time, weekday)] = 0

# initializing how much each of days in week has passed
n_of_days = {}
for i in range(0, 7):
    n_of_days[i] = 0

# collecting overall demand by time of the day and weekday
day = -1    # day within a week
for tid in range(0, N_OF_TIME_SLOTS):   # we must figure out the total value of time slots
    if tid % TIME_SLOTS_WITHIN_DAY == 0:
        day = (day + 1) % 7
        n_of_days[day] += 1
    for cid in range(0, N_OF_CLUSTERS):
        demand = demandCache.get_demand(tid, cid)
        data[(cid, tid % TIME_SLOTS_WITHIN_DAY, day)] += demand
    print(str(tid))

# here we get the means and write them in the file
f = open("means2.txt", "w")
for day_time in range(0, TIME_SLOTS_WITHIN_DAY):
    for weekday in range(0, 7):
        for cid in range(0, N_OF_CLUSTERS):
            data[(cid, day_time, weekday)] /= n_of_days[weekday]
            f.write(str(cid) + " " + str(day_time) + " " + str(weekday) + " " + str(data[(cid, day_time, weekday)]) + "\n")
f.close()

# here we are going to evaluate mean prediction on the following day (first day which data isn't taken
# for the training set) -----> E.G. Prediction for tomorrow
predictions = {}
cnt = 0
next_day = (day + 1) % 7
for tid in range(N_OF_TIME_SLOTS, N_OF_TIME_SLOTS + TIME_SLOTS_WITHIN_DAY):
    for cid in range(0, N_OF_CLUSTERS):
        demand = demandCache.get_demand(tid, cid)
        prediction = data[(cid, tid % TIME_SLOTS_WITHIN_DAY, next_day)]
    if abs(demand - prediction) <= THRESHOLD:
        cnt += 1
    predictions[tid % TIME_SLOTS_WITHIN_DAY] = demand - prediction


print("Model is right in " + str(cnt/TIME_SLOTS_WITHIN_DAY) + "% of cases.")

# print(predictions)
