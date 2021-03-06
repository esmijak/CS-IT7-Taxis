#!/usr/bin/python3

from pyspark.sql import *
from get_features import *
from pyspark.ml.evaluation import RegressionEvaluator
import os
import get_features

from numpy import arange

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"

N_OF_CLUSTERS = 358  # number of clusters for which mean is being calculated

"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144  # day is divided into that number of slots
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 29
FIRST_DAY_DAY_OF_WEEK = 3  # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
FIRST_DAY_OF_TEST = 0
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY  # number of days used for the learning
N_OF_TIME_SLOTS_TRAIN = N_DAYS_TRAIN * TIME_SLOTS_WITHIN_DAY  # number of time slots that are being used for training
N_DAYS_TEST = N_DAYS_JUN
N_OF_TIME_SLOTS_TEST = N_DAYS_TEST * TIME_SLOTS_WITHIN_DAY

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

HORIZONTAL_SLOTS = 25
VERTICAL_SLOTS = 25

rmses = []
r2s = []


# Get count of weekdays in the training set
WEEKDAY_DAYS_COUNT = []

# Add "whole weeks"
for i in range(7):
    WEEKDAY_DAYS_COUNT.append((N_DAYS_TRAIN - (7 - FIRST_DAY_DAY_OF_WEEK)) / 7)

# Add days before the beginning of the first whole week
for i in range(FIRST_DAY_DAY_OF_WEEK, 7):
    WEEKDAY_DAYS_COUNT[i] += 1

# Add days elapsed after the last whole week
for i in range((N_DAYS_TRAIN - (7 - FIRST_DAY_DAY_OF_WEEK)) % 7):
    WEEKDAY_DAYS_COUNT[i] += 1


# Get count of weekdays in the testing set
TEST_WEEKDAY_DAYS_COUNT = []
for i in range(7):
    TEST_WEEKDAY_DAYS_COUNT.append((N_DAYS_TEST - (7 - FIRST_DAY_OF_TEST)) / 7)
for i in range(FIRST_DAY_OF_TEST, 7):
    TEST_WEEKDAY_DAYS_COUNT[i] += 1
for i in range((N_DAYS_TEST - (7 - FIRST_DAY_OF_TEST)) % 7):
    TEST_WEEKDAY_DAYS_COUNT[i] += 1


registerTable(sqlCtx, Table.FINAL_DATA)


def means_weekday_hour():
    rmses.clear()
    r2s.clear()

    weekday_hours_means = [[0 for i in range(24)] for j in range(7)]

    for week_day in range(7):
        for hour in range(24):
            data = spark.sql("SELECT SUM(amount) AS amount "
                             "FROM final_data" +
                             " WHERE week < 23 AND day_of_week = " + str(week_day + 3) + " AND hour = " + str(hour) +
                             " AND origin < " + str(N_OF_CLUSTERS)
                             ).collect()
            if data[0]["amount"] is None:
                amount = 0
            else:
                amount = data[0]["amount"]
            weekday_hours_means[week_day][hour] = 1.0 * amount / (WEEKDAY_DAYS_COUNT[week_day] * 6 * N_OF_CLUSTERS)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            for hour in range(24):
                data = spark.sql("SELECT amount "
                                 "FROM final_data " +
                                 "WHERE week >= 23 AND day = " + str(day) + " AND hour=" + str(hour) +
                                 " AND origin = " + str(cid)).collect()
                amounts = list(map(lambda row: row['amount'], data))
                while len(amounts) < 6:
                    amounts.append(0)
                test_labels.extend(amounts)

                test_features.extend([(week_day, hour) for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        predictions = []
        for features in test_features:
            prediction = weekday_hours_means[features[0]][features[1]]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("means_weekday_hour")


def means_clusters_weekday_hour():
    rmses.clear()
    r2s.clear()

    cluster_weekday_hours_means = [[[0 for i in range(24)] for j in range(7)] for k in range(N_OF_CLUSTERS)]

    for cid in range(N_OF_CLUSTERS):
        for week_day in range(7):
            for hour in range(24):
                data = spark.sql("SELECT SUM(amount) AS amount "
                                 "FROM final_data" +
                                 " WHERE week < 23 AND day_of_week = " + str(week_day + 3) + " AND hour = " + str(hour)
                                 + " AND origin = " + str(cid)
                                 ).collect()
                if data[0]["amount"] is None:
                    amount = 0
                else:
                    amount = data[0]["amount"]
                cluster_weekday_hours_means[cid][week_day][hour] = 1.0 * amount / (WEEKDAY_DAYS_COUNT[week_day] * 6)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            for hour in range(24):
                data = spark.sql("SELECT amount "
                                 "FROM final_data " +
                                 "WHERE week >= 23 AND day = " + str(day) + " AND hour=" + str(hour) +
                                 " AND origin = " + str(cid)).collect()
                amounts = list(map(lambda row: row['amount'], data))
                while len(amounts) < 6:
                    amounts.append(0)
                test_labels.extend(amounts)

                test_features.extend([(week_day, hour) for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        predictions = []
        for features in test_features:
            prediction = cluster_weekday_hours_means[cid][features[0]][features[1]]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("means_clusters_weekday_hour")


def means_grid_weekday_hour():
    rmses.clear()
    r2s.clear()

    grid = spark.read.parquet(hadoopify('grids/final_features_grid'))
    grid.registerTempTable('grid')

    for x in arange(HORIZONTAL_SLOTS):
        for y in arange(VERTICAL_SLOTS):
            (train_features, train_labels), (test_features, test_labels) = get_features.get_features_for_grid(spark, x, y)

            weekday_hours_means = [[0 for i in range(24)] for j in range(7)]
            train_labels_features = list(zip(train_labels, train_features))

            for week_day in range(7):

                for hour in range(24):
                    weekday_hour_sum = list(filter(lambda row: row[1].day_of_week == week_day + 3 and row[1].hour == hour, train_labels_features))
                    weekday_hour_sum = sum(list(map(lambda row: row[0], weekday_hour_sum)))

                    weekday_hours_means[week_day][hour] = (float(weekday_hour_sum) / WEEKDAY_DAYS_COUNT[week_day])

            test_labels = []
            test_features = []

            week_day = FIRST_DAY_OF_TEST
            for day in range(1, N_DAYS_TEST + 1):
                for hour in range(24):
                    data = spark.sql("SELECT amount "
                                     "FROM grid " +
                                     "WHERE week >= 23 AND day = " + str(day) + " AND hour=" + str(hour) +
                                     " AND pickup_lat_slot = " + str(x) + " AND pickup_long_slot = " + str(y)).collect()

                    amounts = list(map(lambda row: row['amount'], data))
                    while len(amounts) < 6:
                        amounts.append(0)
                    test_labels.extend(amounts)

                    test_features.extend([(week_day, hour) for i in range(len(amounts))])

                week_day += 1
                if week_day == 7:
                    week_day = 0

            predictions = []
            for features in test_features:
                prediction = weekday_hours_means[features[0]][features[1]]
                predictions.append(prediction)
            eval(predictions, test_labels)

    write_to_files("grid_weekday_hour", "grid")


def eval(predicted, actual):
    # merge actual_demand and predictions to dataframe for RegressionEvaluator
    dataframe = spark.createDataFrame(zip(predicted, actual), ["prediction", "demand"])

    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
    rmse = evaluatorRMSE.evaluate(dataframe)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    """ Evaluation r2 : """
    evaluatorR2 = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(dataframe)
    print("R Squared Error (R2) on test data = %g" % r2)

    rmses.append(rmse)
    r2s.append(r2)


def write_to_files(method, partition = "cluster"):
    """ Writing the errors in the files : """
    file = open(method + "_rmse.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")

    for errorIndex in range(len(rmses)):
        file.write("RMSE for " + partition + " " + str(errorIndex) + " is " + str(rmses[errorIndex]) + "\n")
    file.close()

    file = open(method + "_r2.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(len(r2s)):
        file.write("R2 for " + partition + " " + str(errorIndex) + " is " + str(r2s[errorIndex]) + "\n")
    file.close()


means_grid_weekday_hour()
