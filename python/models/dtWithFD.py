import numpy as np
from pyspark.sql import *
from schemas import *
from demand_cache import invDemandCache
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import datetime


"""Cluster related constant: """
N_OF_CLUSTERS = 1000   # number of clusters used : all of them

"""Time related constant: """
TIME_SLOTS_WITHIN_DAY = 144     # day is divided into that number of slots
N_DAYS_JAN = 31
N_DAYS_FEB = 28
N_DAYS_MAR = 31
N_DAYS_APR = 30
N_DAYS_MAY = 31
N_DAYS_JUN = 30
FIRST_DAY_DAY_OF_WEEK = 3   # which day of the week was the first day of the year 2015 (0 - Monday, 1 - Tuesday, etc.)
N_DAYS_TRAIN = N_DAYS_JAN + N_DAYS_FEB + N_DAYS_MAR + N_DAYS_APR + N_DAYS_MAY # number of days used for the learning
N_OF_TIME_SLOTS_TRAIN = N_DAYS_TRAIN * TIME_SLOTS_WITHIN_DAY # number of time slots that are being used for training
N_DAYS_TEST = N_DAYS_JUN
N_OF_TIME_SLOTS_TEST =  N_DAYS_TEST * TIME_SLOTS_WITHIN_DAY





"""Spark initialization: """
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

featuresTable = loadDataFrame(sqlCtx, Table.FINAL_DATA)
slotsTable = loadDataFrame(sqlCtx, Table.TIME_SLOTS)
slotsTableCol = slotsTable.collect()
registerTable(sqlCtx, Table.TIME_SLOTS)


errorsRMSE = []
errorsR2 = []

def get_time_slot_per_day(hour, minute):
    nb_minute_slot = minute / 10
    time_slot_per_hour = 6
    slot_nb_day = hour * time_slot_per_hour + nb_minute_slot
    return slot_nb_day

def getSlotInfo(slot_nb) :
    df_slot = slotsTable[slotsTable.id == slot_nb].select('from').collect()
    test = df_slot[0]
    day_of_week = test[0].weekday()
    day = test[0].day
    week = test[0].isocalendar()[1]
    hour = test[0].hour
    minute = test[0].minute
    time_of_day_code = get_time_slot_per_day(hour, minute)
    return week, day, day_of_week, time_of_day_code, hour, minute

for curCluster in range (N_OF_CLUSTERS):
#for curCluster in cc: to test a specific cluster
    print('current cluster number is: ', curCluster)

    """ df is data frame containing the time slot id, demand for the current cluster"""
    df_for_one_cluster = featuresTable[featuresTable.origin == curCluster].select('month','week','day','day_of_week','time_of_day_code','hour','minute','origin','amount','pickup_timeslot_id')
    #df_for_one_cluster.show()

    print("data downloaded!")

    #demandListDict = df_for_one_cluster.collect()

    rows_training = []
    rows_testing = []
    demandCount = 0
    TOTAL_SLOTS_FOR_LOOP = N_OF_TIME_SLOTS_TEST + N_OF_TIME_SLOTS_TRAIN
    print('total time slots in loop: ', TOTAL_SLOTS_FOR_LOOP)
    for curSlotNb in range (TOTAL_SLOTS_FOR_LOOP) :
        if curSlotNb % 1000 == 0 :
            print('cur slot: ', curSlotNb)
        df_for_one_cluster_for_one_ts = df_for_one_cluster[featuresTable.pickup_timeslot_id == curSlotNb].select('week',
                                                                                                       'day',
                                                                                                       'day_of_week',
                                                                                                       'time_of_day_code',
                                                                                                       'hour', 'minute',
                                                                                                       'origin',
                                                                                                       'amount',
                                                                                                       'pickup_timeslot_id')
        df_for_one_cluster_for_one_ts_col = df_for_one_cluster_for_one_ts.collect()

        if len(df_for_one_cluster_for_one_ts_col) < 1 :
            week, day, day_of_week, time_of_day_code, hour, minute = getSlotInfo(curSlotNb)
            demand = 0

        else :
            """Compute the time slot information: """
            weekday = df_for_one_cluster_for_one_ts_col[0].day_of_week
            day = df_for_one_cluster_for_one_ts[0].day
            week_nb =  df_for_one_cluster_for_one_ts[0].week
            hour = df_for_one_cluster_for_one_ts[0].hour
            minute = df_for_one_cluster_for_one_ts[0].minute
            demand = df_for_one_cluster_for_one_ts[0].amount
            slot_nb_day = df_for_one_cluster_for_one_ts[0].time_of_day_code

        if (len(rows_training) < N_OF_TIME_SLOTS_TRAIN):
            rows_training.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))
        else:
            rows_testing.append((slot_nb_day, weekday, day, week_nb, hour, minute, demand))


    print('train rows len: ', len(rows_training), 'test rows len: ', len(rows_testing))

    """Create 2 dataframes corresponding to the training and testing set previously computed: """
    df_training = spark.createDataFrame(rows_training, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])
    df_testing = spark.createDataFrame(rows_testing, ["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute", "demand"])


    assembler = VectorAssembler(inputCols=["slot_id", "day_of_week", "day_of_month", "week_nb", "hour", "minute"], outputCol='features')
    output_training = assembler.transform(df_training)
    output_testing = assembler.transform(df_testing)

    final_data_training = output_training.select('features', 'demand')
    final_data_testing = output_testing.select('features', 'demand')


    final_data_training.describe().show()
    final_data_testing.describe().show()



    """  Model and predictions : """
    decisionTree = DecisionTreeRegressor(labelCol='demand', maxDepth=3)
    dt_model = decisionTree.fit(final_data_training)
    predictions = dt_model.transform(final_data_testing)
    #print("Decision tree model max depth = %g" % decisionTree.getMaxDepth())
    #print(dt_model.toDebugString)


    """ Evaluation rmse : """
    evaluatorRMSE = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
    rmse = evaluatorRMSE.evaluate(predictions)
    errorsRMSE.append(rmse)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    evaluatorR2 = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="r2")
    r2 = evaluatorR2.evaluate(predictions)
    errorsR2.append(r2)
    print("R Squared Error (R2) on test data = %g" % r2)


""" Writing the errors in the files : """
file = open("decision_tree_rmse.txt", "w")
file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("RMSE for cluster " + str(errorIndex) + " is " + str(errorsRMSE[errorIndex]) + "\n")
file.close()

file = open("decision_tree_r2.txt", "w")
file.write("Training set contains " + str(N_DAYS_TRAIN) + " days i.e. "+ str(N_OF_TIME_SLOTS_TRAIN) + " time slots \nTest set contains "+ str(N_DAYS_TEST)+ " days i.e. "+ str(N_OF_TIME_SLOTS_TEST) + " time slots \n")
for errorIndex in range(N_OF_CLUSTERS):
    file.write("R2 for cluster " + str(errorIndex) + " is " + str(errorsR2[errorIndex]) + "\n")
file.close()
