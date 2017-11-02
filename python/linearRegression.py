#!/usr/bin/python3

from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from schemas import *

spark = SparkSession.builder.master('spark://172.25.24.242:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)

dataDFR = loadDataFrame(sqlCtx, Table.DEMAND_SAMPLE)
print(dataDFR)

assembler = VectorAssembler(inputCols=["pickup_cid", "pickup_timeslot_id"], outputCol="features")

output = assembler.transform(dataDFR)
output.printSchema()
final_data = output.select('features', 'demand')
final_data.show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])
train_data.describe().show()
test_data.describe().show()

lr = LinearRegression(labelCol='demand')
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)
test_results.residuals.show()

er1 = test_results.rootMeanSquaredError
er2 = test_results.r2
final_data.describe().show()

unlabeled_data = test_data.select('features')
unlabeled_data.show()
predictions = lr_model.transform(unlabeled_data)
predictions.show()
