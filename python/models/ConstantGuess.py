from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from schemas import *
import pandas as pd
import numpy as np
from operator import add

""" fetching the Demand sample table : """
spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

data_formated = loadDataFrame(sqlCtx, Table.DEMAND_SAMPLE)
#data_formated.show()


""" formating the data in a form accepted by the ml library : """
assembler = VectorAssembler(inputCols=['pickup_timeslot_id', 'pickup_cid'], outputCol='features')
output = assembler.transform(data_formated)
final_data = output.select('features', 'demand')
#final_data.show()

""" splitting randomly the data into test and learning sets : """
train_data, test_data = final_data.randomSplit([0.7, 0.3])

""" constant guess : """
test_data.registerTempTable("test_data")
size = test_data.count()

def constant_guess(constant):
    error = 0
    error = spark.sql("SELECT SUM(POWER({} - demand, 2)) AS error FROM test_data".format(constant)).collect()[0][0]
    error /= size
    error = np.sqrt(error)
    print("RMSE for guessing {}: {}\n".format(constant, error))

constant_guess(0)
constant_guess(1)