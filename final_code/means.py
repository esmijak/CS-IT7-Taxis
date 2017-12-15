#!/usr/bin/python3

from pyspark.sql import *
from get_features import *
from pyspark.ml.evaluation import RegressionEvaluator
import os
import get_features
from numpy import arange

"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144  # day is divided into that number of slots
TIME_SLOTS_WITHIN_HOUR = 6
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 29
FIRST_DAY_DAY_OF_WEEK = 3  # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
TRAIN_WEEKS_COUNT = 23
FIRST_DAY_OF_TEST = 0
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY  # number of days used for the learning
N_OF_TIME_SLOTS_TRAIN = N_DAYS_TRAIN * TIME_SLOTS_WITHIN_DAY  # number of time slots that are being used for training
N_DAYS_TEST = N_DAYS_JUN
N_OF_TIME_SLOTS_TEST = N_DAYS_TEST * TIME_SLOTS_WITHIN_DAY
WEEK_DAYS_SHIFT = 3  # Index of week days in the DB is shifted with this number

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

# Get count of weekdays in the testing set as well
TEST_WEEKDAY_DAYS_COUNT = []
for i in range(7):
    TEST_WEEKDAY_DAYS_COUNT.append((N_DAYS_TEST - (7 - FIRST_DAY_OF_TEST)) / 7)
for i in range(FIRST_DAY_OF_TEST, 7):
    TEST_WEEKDAY_DAYS_COUNT[i] += 1
for i in range((N_DAYS_TEST - (7 - FIRST_DAY_OF_TEST)) % 7):
    TEST_WEEKDAY_DAYS_COUNT[i] += 1

"""Location related constants"""
HORIZONTAL_SLOTS = 25  # number of slots in grid
VERTICAL_SLOTS = 25  # number of slots in grid

N_OF_CLUSTERS = 358  # number of clusters for which mean is being calculated (358 for all)


"""Setting up Spark"""
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
registerTable(sqlCtx, Table.FINAL_DATA)


def means_weekday_hour():
    # Init model
    weekday_hours_means = [[0 for i in range(24)] for j in range(7)]

    # Train model (compute mean for each week_day and hour combination)
    for week_day in range(7):
        for hour in range(24):
            data = spark.sql("SELECT SUM(amount) AS amount"
                             " FROM final_data" +
                             " WHERE week < " + str(TRAIN_WEEKS_COUNT) +
                             " AND day_of_week = " + str(week_day + WEEK_DAYS_SHIFT) +
                             " AND hour = " + str(hour) +
                             " AND origin < " + str(N_OF_CLUSTERS)
                             ).collect()
            if data[0]["amount"] is None:
                amount = 0
            else:
                amount = data[0]["amount"]
            weekday_hours_means[week_day][hour] = float(amount) / (WEEKDAY_DAYS_COUNT[week_day] * TIME_SLOTS_WITHIN_HOUR * N_OF_CLUSTERS)

    # Variables used for evaluation
    rmses = []
    r2s = []

    all_predictions = []
    all_actuals = []

    # Evaluate for each cluster independently
    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        # Retrieving all combinations of weekday and hour manually, because 0 demand entries are not present
        # in the DB and therefore they have to be filled in
        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            for hour in range(24):
                data = spark.sql("SELECT amount "
                                 " FROM final_data" +
                                 " WHERE week >= " + str(TRAIN_WEEKS_COUNT) +
                                 " AND day = " + str(day + WEEK_DAYS_SHIFT) + " AND hour=" + str(hour) +
                                 " AND origin = " + str(cid)
                                 ).collect()
                amounts = list(map(lambda row: row['amount'], data))
                while len(amounts) < TIME_SLOTS_WITHIN_HOUR:
                    amounts.append(0)
                test_labels.extend(amounts)

                test_features.extend([(week_day, hour) for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        # Predict (retrieve relevant week day and hour average)
        predictions = []
        for features in test_features:
            prediction = weekday_hours_means[features[0]][features[1]]
            predictions.append(prediction)

        # Evaluate
        rmse, r2 = eval(predictions, test_labels)
        rmses.append(rmse)
        r2s.append(r2)

        # Accumulate all predictions and actual results for overall evaluation
        all_predictions.extend(predictions)
        all_actuals.extend(test_labels)

    overall_rmse, overall_r2 = eval(all_predictions, all_actuals)
    clusters_write_to_files("means_weekday_hour", overall_rmse, overall_r2, rmses, r2s)


def means_clusters_weekday_hour():
    # Init model
    cluster_weekday_hours_means = [[[0 for i in range(24)] for j in range(7)] for k in range(N_OF_CLUSTERS)]

    # Train model (compute mean for each cluster, week_day, and hour combination)
    for cid in range(N_OF_CLUSTERS):
        for week_day in range(7):
            for hour in range(24):
                data = spark.sql("SELECT SUM(amount) AS amount"
                                 " FROM final_data" +
                                 " WHERE week < " + str(TRAIN_WEEKS_COUNT) +
                                 " AND day_of_week = " + str(week_day + WEEK_DAYS_SHIFT) +
                                 " AND hour = " + str(hour) +
                                 " AND origin = " + str(cid)
                                 ).collect()
                if data[0]["amount"] is None:
                    amount = 0
                else:
                    amount = data[0]["amount"]
                cluster_weekday_hours_means[cid][week_day][hour] = float(amount) / (WEEKDAY_DAYS_COUNT[week_day] * TIME_SLOTS_WITHIN_HOUR)

    # Variables used for evaluation
    rmses = []
    r2s = []

    all_predictions = []
    all_actuals = []

    # Evaluate for each cluster independently
    for cid in range(N_OF_CLUSTERS):
        test_labels = []
        test_features = []

        # Retrieving all combinations of weekday and hour manually, because 0 demand entries are not present
        # in the DB and therefore they have to be filled in
        week_day = FIRST_DAY_OF_TEST
        for day in range(1, N_DAYS_TEST + 1):
            for hour in range(24):
                data = spark.sql("SELECT amount"
                                 " FROM final_data " +
                                 " WHERE week >= " + str(TRAIN_WEEKS_COUNT) +
                                 " AND day = " + str(day) + " AND hour=" + str(hour) +
                                 " AND origin = " + str(cid)
                                 ).collect()
                amounts = list(map(lambda row: row['amount'], data))
                while len(amounts) < TIME_SLOTS_WITHIN_HOUR:
                    amounts.append(0)
                test_labels.extend(amounts)

                test_features.extend([(week_day, hour) for i in range(len(amounts))])

            week_day += 1
            if week_day == 7:
                week_day = 0

        # Predict (retrieve relevant cluster, week_day, and hour average)
        predictions = []
        for features in test_features:
            prediction = cluster_weekday_hours_means[cid][features[0]][features[1]]
            predictions.append(prediction)

        # Evaluate
        rmse, r2 = eval(predictions, test_labels)
        rmses.append(rmse)
        r2s.append(r2)

        # Accumulate all predictions and actual results for overall evaluation
        all_predictions.extend(predictions)
        all_actuals.extend(test_labels)

    overall_rmse, overall_r2 = eval(all_predictions, all_actuals)
    clusters_write_to_files("means_clusters_weekday_hour", overall_rmse, overall_r2, rmses, r2s)


def means_grid_weekday_hour():
    # Variables used for evaluation
    rmses = [[0 for i in range(VERTICAL_SLOTS)] for j in range(HORIZONTAL_SLOTS)]
    r2s = [[0 for i in range(VERTICAL_SLOTS)] for j in range(HORIZONTAL_SLOTS)]

    all_predictions = []
    all_actuals = []

    # Train and evaluate for each piece of the grid in one cycle iteration
    for x in arange(HORIZONTAL_SLOTS):
        for y in arange(VERTICAL_SLOTS):
            # Retrieve grid data from the cache
            (train_features, train_labels), (test_features, test_labels) = get_features.get_features_for_grid(spark, x, y)

            # Init model
            weekday_hours_means = [[0 for i in range(24)] for j in range(7)]
            train_labels_features = list(zip(train_labels, train_features))

            # Train model (compute mean for week_day and hour combination)
            for week_day in range(7):
                for hour in range(24):
                    weekday_hour_sum = list(filter(lambda row: row[1].day_of_week == week_day + WEEK_DAYS_SHIFT and row[1].hour == hour, train_labels_features))
                    weekday_hour_sum = sum(list(map(lambda row: row[0], weekday_hour_sum)))

                    weekday_hours_means[week_day][hour] = (float(weekday_hour_sum) / WEEKDAY_DAYS_COUNT[week_day])

            # Predict (retrieve relevant week_day and hour average)
            predictions = []
            for features in test_features:
                prediction = weekday_hours_means[features["day_of_week"] - WEEK_DAYS_SHIFT][features["hour"]]
                predictions.append(prediction)

            rmse, r2 = eval(predictions, test_labels)
            rmses[x][y] = rmse
            r2s[x][y] = r2

            # Accumulate all predictions and actual results for overall evaluation
            all_predictions.extend(predictions)
            all_actuals.extend(test_labels)

    overall_rmse, overall_r2 = eval(all_predictions, all_actuals)
    grid_write_to_files("grid_weekday_hour", overall_rmse, overall_r2, rmses, r2s)


def eval(predicted, actual):
    # merge actual_demandoverall_rmse, overall_r2, and predictions to dataframe for RegressionEvaluator
    dataframe = spark.createDataFrame(zip(predicted, actual), ["prediction", "demand"])

    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
    rmse = evaluatorRMSE.evaluate(dataframe)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    """ Evaluation r2 : """
    evaluatorR2 = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(dataframe)
    print("R Squared Error (R2) on test data = %g" % r2)

    return rmse, r2


def clusters_write_to_files(method, overall_rmse, overall_r2, rmses, r2s):
    """ Writing the errors in the files : """
    file = open(method + "_rmse.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")

    file.write("Overall rmse is: " + str(overall_rmse))
    file.write("Overal r2 is: " + str(overall_r2))

    for errorIndex in range(len(rmses)):
        file.write("RMSE for cluster " + str(errorIndex) + " is " + str(rmses[errorIndex]) + "\n")
    file.close()

    file = open(method + "_r2.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for errorIndex in range(len(r2s)):
        file.write("R2 for cluster " + str(errorIndex) + " is " + str(r2s[errorIndex]) + "\n")
    file.close()


def grid_write_to_files(method, overall_rmse, overall_r2, rmses, r2s):
    """ Writing the errors in the files : """
    file = open(method + "_rmse.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")

    file.write("Overall rmse is: " + str(overall_rmse))
    file.write("Overal r2 is: " + str(overall_r2))

    for x in range(HORIZONTAL_SLOTS):
        for y in range(VERTICAL_SLOTS):
            file.write("RMSE for grid :  (" + str(x) + ', ' + str(y) + ") is " + str(rmses[x][y]) + "\n")
    file.close()

    file = open(method + "_r2.txt", "w")
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains " + str(N_DAYS_TEST) + " days i.e. " + str(
        N_OF_TIME_SLOTS_TEST) + " time slots \n")
    for x in range(HORIZONTAL_SLOTS):
        for y in range(VERTICAL_SLOTS):
            file.write("R2 for grid :  (" + str(x) + ', ' + str(y) + ") is " + str(r2s[x][y]) + "\n")
    file.close()


# Call model training with evaluation here
means_weekday_hour()
means_grid_weekday_hour()
means_clusters_weekday_hour()
