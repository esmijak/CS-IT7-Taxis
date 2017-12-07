#!/usr/bin/python3

from schemas import *
from pyspark.sql import *
from feature_cache import invDemandCache
from feature_cache import DemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

"""Cluster related constant: """
N_OF_CLUSTERS = 358   # number of clusters used : all of them
"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144    # day is divided into that number of slots
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 30
FIRST_DAY_DAY_OF_WEEK = 3   # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY # number of days used for the learning
#N_DAYS_TRAIN = 1
N_OF_TIME_SLOTS_TRAIN = N_DAYS_TRAIN * TIME_SLOTS_WITHIN_DAY # number of time slots that are being used for training
N_DAYS_TEST = N_DAYS_JUN
#N_DAYS_TEST = 1
N_OF_TIME_SLOTS_TEST =  N_DAYS_TEST * TIME_SLOTS_WITHIN_DAY


spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)
slotsTable = loadDataFrame(sqlCtx, Table.TIME_SLOTS)
slotsTableCol = slotsTable.collect()
clustTable = loadDataFrame(sqlCtx, Table.CLUSTER_DATA)
clustTableCol = clustTable.collect()



invDemandCache.init(spark, sqlCtx)

def get_time_slot_per_day(hour, minute):
    nb_minute_slot = minute / 10
    time_slot_per_hour = 6
    slot_nb_day = hour * time_slot_per_hour + nb_minute_slot
    return slot_nb_day

def getSlotInfo(slot_nb) :
    df_slot = slotsTable[slotsTable.id == slot_nb].select('from').collect()
    test = df_slot[0]
    day_of_week = int(test[0].weekday())
    day = int(test[0].day)
    week = int(test[0].isocalendar()[1])
    hour = int(test[0].hour)
    minute = int(test[0].minute)
    time_of_day_code = int(get_time_slot_per_day(hour, minute))
    return week, day, day_of_week, time_of_day_code, hour, minute

def getClusterInfo(cluster_nb) : #TODO retrieve the cluster info !!!
    df_clust = clustTable[clustTable.ride_id == cluster_nb]
    info = df_clust.take(1)[0]
    cetroid_long = info['cetroid_long']
    centroid_lat = info['centroid_lat']
    isManhattan = 0
    isAirport = 0
    if cetroid_long > -74.025 and cetroid_long < -73.975 and centroid_lat > 40.705 and centroid_lat < 40.75 :
        isManhattan = 1
    elif cetroid_long > -73.81 and cetroid_long < -73.77 and centroid_lat > 40.64 and centroid_lat < 40.663 :
        isAirport = 1
    return isManhattan, isAirport

TOTAL_SLOTS_FOR_LOOP = N_OF_TIME_SLOTS_TEST + N_OF_TIME_SLOTS_TRAIN


def extract_feature(curFeature) :
    month = curFeature['month']
    week = curFeature['week']
    day = curFeature['day']
    time_of_day_code = curFeature['time_of_day_code']
    day_of_week = curFeature['day_of_week']
    hour = curFeature['hour']
    origin = curFeature['origin']
    is_manhattan = curFeature['is_manhattan']
    is_airport = curFeature['is_airport']
    amount = curFeature['amount']
    pickup_timeslot_id = curFeature['pickup_timeslot_id']
    return time_of_day_code, day_of_week, day, week, hour, is_manhattan, is_airport, amount

def get_data(start_time, end_time, cluster):
    print('call')
    rows = []
    for tid in range(start_time, end_time):
        if tid % 50 == 0 :
            print(tid)
        features = invDemandCache.get_demand(tid, cluster)
        if features.count() > 0 :
            curFeature = features.take(1)[0]
            time_of_day_code, day_of_week, day, week, hour, is_manhattan, is_airport, amount = extract_feature(curFeature)
        else:
            week, day, day_of_week, time_of_day_code, hour, minute = getSlotInfo(cluster)
            is_manhattan, is_airport = getClusterInfo(cluster)
            amount = 0

        rows.append((time_of_day_code, day_of_week, day, week, hour, is_manhattan, is_airport, amount))

    print('train rows len: ', len(rows))

    df = spark.createDataFrame(rows, ["time_of_day_code", "day_of_week", "day", "week", "hour", "is_manhattan", "is_airport", "amount"])
    assembler = VectorAssembler(
        inputCols=['time_of_day_code', 'day_of_week', 'day', 'week', 'hour', 'is_manhattan', 'is_airport'],
        outputCol='features')
    output = assembler.transform(df)
    return output.select('features', 'amount')

file = open("feature_model.txt", "w")
#errors = []
for cid in range(0, N_OF_CLUSTERS):
    final_data_training = get_data(0, N_OF_TIME_SLOTS_TRAIN, cid)
    final_data_testing = get_data(N_OF_TIME_SLOTS_TRAIN, N_OF_TIME_SLOTS_TRAIN + N_OF_TIME_SLOTS_TEST, cid)


    final_data_training.describe().show()
    final_data_testing.describe().show()

    """  Model and predictions : """
    decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
    dt_model = decisionTree.fit(final_data_training)
    predictions = dt_model.transform(final_data_testing)

    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
    errorRMSE = evaluatorRMSE.evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = %g" % errorRMSE)

    evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
    errorR2 = evaluatorR2.evaluate(predictions)
    print("R Squared Error (R2) on test data = %g" % errorR2)

    """ Writing the errors in the files : """
    file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
    file.write('cluster number :'+ str(cid))
    file.write("\nRMSE is " + str(errorRMSE) + "\n")
    file.write("R2 is " + str(errorR2)  + "\n")

#
#    lr_model = lr.fit(train_data)
#    test_results = lr_model.evaluate(test_data)
#    error = test_results.rootMeanSquaredError
#    errors.append((cid, error))


def get_data_all_clust(start_time, end_time):
    print('call')
    rows = []
    for cid in range (N_OF_CLUSTERS) :
        for tid in range(start_time, end_time):
            if tid % 50 == 0 :
                print(tid)
            features = invDemandCache.get_demand(tid, cid)
            if features.count() > 0 :
                curFeature = features.take(1)[0]
                time_of_day_code, day_of_week, day, week, hour, is_manhattan, is_airport, amount = extract_feature(curFeature)
            else:
                week, day, day_of_week, time_of_day_code, hour, minute = getSlotInfo(cid)
                is_manhattan, is_airport = getClusterInfo(cid)
                amount = 0
            rows.append((cid, time_of_day_code, day_of_week, day, week, hour, is_manhattan, is_airport, amount))

    df = spark.createDataFrame(rows, ['cid', 'time_of_day_code', 'day_of_week', 'day', 'week', 'hour', 'is_manhattan', 'is_airport', 'amount'])
    assembler = VectorAssembler(
        inputCols=['cid', 'time_of_day_code', 'day_of_week', 'day', 'week', 'hour', 'is_manhattan', 'is_airport'],
        outputCol='features')
    output = assembler.transform(df)
    return output.select('features', 'amount')

final_data_training = get_data_all_clust(0, N_OF_TIME_SLOTS_TRAIN)
final_data_testing = get_data_all_clust(N_OF_TIME_SLOTS_TRAIN, N_OF_TIME_SLOTS_TRAIN + N_OF_TIME_SLOTS_TEST)


final_data_training.describe().show()
final_data_testing.describe().show()

"""  Model and predictions : """
decisionTree = DecisionTreeRegressor(labelCol='amount', maxDepth=3)
dt_model = decisionTree.fit(final_data_training)
predictions = dt_model.transform(final_data_testing)

""" Evaluation rmse : """
evaluatorRMSE = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="rmse")
errorRMSE = evaluatorRMSE.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % errorRMSE)

evaluatorR2 = RegressionEvaluator(labelCol="amount", predictionCol="prediction", metricName="r2")
errorR2 = evaluatorR2.evaluate(predictions)
print("R Squared Error (R2) on test data = %g" % errorR2)

""" Writing the errors in the files : """
file.write("\n\nFor all clusters:")
file.write("\nRMSE is " + str(errorRMSE) + "\n")
file.write("R2 is " + str(errorR2)  + "\n")

file.close()