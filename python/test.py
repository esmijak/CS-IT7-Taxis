from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from  pyspark.ml.regression import  DecisionTreeRegressor
from schemas import *
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import datetime

""" fetching the Demand sample table: """
spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

data_formated = loadDataFrame(sqlCtx, Table.DEMAND_SAMPLE)
slots = loadDataFrame(sqlCtx, Table.TIME_SLOTS).collect()
slot_count = len(slots)

""" Reorganizing the time slots: """
time_per_slot = (60 * 60 * 24 * 180) / slot_count
start_time = datetime.datetime(2015, 1, 1, 0).timestamp()
def find_slot(time):
    return int((time.timestamp() - start_time) / time_per_slot)

def find_date_time (time_slot) :
    time_stamp = (time_slot * time_per_slot) + start_time
    return datetime.datetime.fromtimestamp(time_stamp)

""" Contructing a table with the slot id and the date information: """
slot_id_info = np.zeros((slot_count, 7)) # slot id,weekday, day, month, year, hour, minute
for instance in slots :
    slot_nb = find_slot(instance[0])
    slot_id_info[slot_nb, 0] = slot_nb
    slot_id_info[slot_nb, 1] = instance[0].weekday()
    slot_id_info[slot_nb, 2] = instance[0].day
    slot_id_info[slot_nb, 3] = instance[0].month
    slot_id_info[slot_nb, 4] = instance[0].year
    slot_id_info[slot_nb, 5] = instance[0].hour
    slot_id_info[slot_nb, 6] = instance[0].minute

""" adding slot info to the formated data:"""


""" formating the data in a form accepted by the ml library : """
assembler = VectorAssembler(inputCols=['pickup_timeslot_id', 'pickup_cid'], outputCol='features')
output = assembler.transform(data_formated)
final_data = output.select('features', 'demand')
#final_data.show()

""" splitting randomly the data into test and learning sets : """
train_data, test_data = final_data.randomSplit([0.7, 0.3])



#train_data.describe().show()
#test_data.describe().show()

"""  Model and predictions : """

decisionTree = DecisionTreeRegressor(labelCol='demand')
dt_model = decisionTree.fit(train_data)
predictions = dt_model.transform(test_data)
""" Evaluation : """

evaluator = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)