#!/usr/bin/python3

from pyspark.sql import *
from get_features import *
from pyspark.ml.evaluation import RegressionEvaluator
import os
import get_features

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"

N_OF_CLUSTERS = 358  # number of clusters for which mean is being calculated

# 358

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
                             " WHERE month < 5 AND day_of_week = " + str(week_day + 3) + " AND hour = " + str(hour) +
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
                                 "WHERE month = 5 AND day = " + str(day) + " AND hour=" + str(hour) +
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


def means_hour():
    rmses.clear()
    r2s.clear()

    hours_means = [0 for i in range(24)]
    for hour in range(24):
        data = spark.sql("SELECT SUM(amount) AS amount "
                         "FROM final_data" +
                         " WHERE month < 5 AND hour = " + str(hour) + " AND origin < " + str(N_OF_CLUSTERS)
                         ).collect()
        if data[0]["amount"] is None:
            amount = 0
        else:
            amount = data[0]["amount"]

        hours_means[hour] = 1.0 * amount / (N_DAYS_TRAIN * 6 * N_OF_CLUSTERS)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        for hour in range(24):
            data = spark.sql("SELECT amount "
                             "FROM final_data " +
                             "WHERE month = 5 AND hour=" + str(hour) +
                             " AND origin = " + str(cid)).collect()
            amounts = list(map(lambda row: row['amount'], data))
            while len(amounts) < 6 * N_DAYS_TEST:
                amounts.append(0)

            test_labels.extend(amounts)
            test_features.extend([hour for i in range(len(amounts))])

        predictions = []
        for features in test_features:
            prediction = hours_means[features]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("means_hour")


def means_weekday():
    rmses.clear()
    r2s.clear()

    weekday_means = [0 for i in range(7)]

    for week_day in range(7):
        data = spark.sql("SELECT SUM(amount) AS amount "
                         "FROM final_data" +
                         " WHERE month < 5 AND day_of_week = " + str(week_day + 3) +
                         " AND origin < " + str(N_OF_CLUSTERS)
                         ).collect()
        if data[0]["amount"] is None:
            amount = 0
        else:
            amount = data[0]["amount"]
        weekday_means[week_day] = 1.0 * amount / (WEEKDAY_DAYS_COUNT[week_day] * 24 * 6 * N_OF_CLUSTERS)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            for hour in range(24):
                data = spark.sql("SELECT amount "
                                 "FROM final_data " +
                                 "WHERE month = 5 AND day = " + str(day) +
                                 " AND origin = " + str(cid)).collect()
                amounts = list(map(lambda row: row['amount'], data))
                while len(amounts) < 6*24:
                    amounts.append(0)
                test_labels.extend(amounts)

                test_features.extend([week_day for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        predictions = []
        for features in test_features:
            prediction = weekday_means[features]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("means_weekday")


def means_clusters_weekday_hour():
    rmses.clear()
    r2s.clear()

    cluster_weekday_hours_means = [[[0 for i in range(24)] for j in range(7)] for k in range(N_OF_CLUSTERS)]

    for cid in range(N_OF_CLUSTERS):
        for week_day in range(7):
            for hour in range(24):
                data = spark.sql("SELECT SUM(amount) AS amount "
                                 "FROM final_data" +
                                 " WHERE month < 5 AND day_of_week = " + str(week_day + 3) + " AND hour = " + str(hour)
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
                                 "WHERE month = 5 AND day = " + str(day) + " AND hour=" + str(hour) +
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


def means_clusters_weekday():
    rmses.clear()
    r2s.clear()

    cluster_weekday_means = [[0 for i in range(7)] for j in range(N_OF_CLUSTERS)]

    for cid in range(N_OF_CLUSTERS):
        for week_day in range(7):
            data = spark.sql("SELECT SUM(amount) AS amount "
                             "FROM final_data " +
                             "WHERE month < 5 AND day_of_week = " + str(week_day + 3) +
                             " AND origin = " + str(cid)
                             ).collect()
            if data[0]["amount"] is None:
                amount = 0
            else:
                amount = data[0]["amount"]
            cluster_weekday_means[cid][week_day] = 1.0 * amount / (WEEKDAY_DAYS_COUNT[week_day] * 6 * 24)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            data = spark.sql("SELECT amount "
                             "FROM final_data " +
                             "WHERE month = 5 AND day = " + str(day) +
                             " AND origin = " + str(cid)).collect()
            amounts = list(map(lambda row: row['amount'], data))
            while len(amounts) < 6*24:
                amounts.append(0)
            test_labels.extend(amounts)

            test_features.extend([week_day for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        predictions = []
        for features in test_features:
            prediction = cluster_weekday_means[cid][features]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("means_clusters_weekday_hour")


def means_clusters_hour():
    rmses.clear()
    r2s.clear()

    cluster_hours_means = [[0 for i in range(24)] for j in range(N_OF_CLUSTERS)]

    for cid in range(N_OF_CLUSTERS):
        for hour in range(24):
            data = spark.sql("SELECT SUM(amount) AS amount "
                             "FROM final_data " +
                             "WHERE month < 5 AND hour = " + str(hour) +
                             " AND origin = " + str(cid)
                             ).collect()
            if data[0]["amount"] is None:
                amount = 0
            else:
                amount = data[0]["amount"]
            cluster_hours_means[cid][hour] = 1.0 * amount / (N_DAYS_TRAIN * 6)

    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        for hour in range(24):
            data = spark.sql("SELECT amount "
                             "FROM final_data " +
                             "WHERE month = 5 AND hour = " + str(hour) +
                             " AND origin = " + str(cid)).collect()
            amounts = list(map(lambda row: row['amount'], data))
            while len(amounts) < N_DAYS_TEST * 6:
                amounts.append(0)
            test_labels.extend(amounts)

            test_features.extend([hour for i in range(len(amounts))])

        predictions = []
        for features in test_features:
            prediction = cluster_hours_means[cid][features]
            predictions.append(prediction)
        eval(predictions, test_labels)

    write_to_files("clusters_hour")


def means_clusters():
    rmses.clear()
    r2s.clear()

    for i in range(N_OF_CLUSTERS):
        (train_features, train_labels), (test_features, test_labels) = get_features_for_cluster(sqlCtx, i)

        mean = sum(train_labels) / N_OF_TIME_SLOTS_TRAIN

        while len(test_labels) < N_OF_TIME_SLOTS_TEST:
            test_labels.append(0)

        predictions = [mean for i in range(len(test_labels))]

        eval(predictions, test_labels)

    write_to_files("clusters")


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


def write_to_files(method):
    """ Writing the errors in the files : """
    file = open(method + "_rmse.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(N_OF_CLUSTERS):
        file.write("RMSE for cluster " + str(errorIndex) + " is " + str(rmses[errorIndex]) + "\n")
    file.close()

    file = open(method + "_r2.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(N_OF_CLUSTERS):
        file.write("R2 for cluster " + str(errorIndex) + " is " + str(r2s[errorIndex]) + "\n")
    file.close()


means_clusters_weekday_hour()
means_clusters()

