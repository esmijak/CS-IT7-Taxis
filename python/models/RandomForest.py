from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from  pyspark.ml.regression import  RandomForestRegressor
from schemas import *
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
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



#train_data.describe().show()
#test_data.describe().show()

"""  Model and predictions : """

randomForest = RandomForestRegressor(labelCol='demand')
rf_model = randomForest.fit(train_data)
print("number of trees :", rf_model.getNumTrees())

predictions = rf_model.transform(test_data)
""" Evaluation : """

evaluator = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)