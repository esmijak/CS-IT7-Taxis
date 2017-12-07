from pyspark.sql import *
from schemas import *
from math import e, log
from numpy import amin
import matplotlib.pyplot as plt

spark = SparkSession.builder.master('spark://csit7-master:7077').getOrCreate()
sqlCtx = SQLContext(spark.sparkContext, spark)


def demandHistogram(**kwargs):
    #df = loadDataFrame(sqlCtx, Table.DEMAND)
    df = spark.read.parquet(hadoopify('clusters/demand500'))
    demand = [row[0] for row in df.select('demand').filter('demand > 0').collect()]
    return demand, plt.hist(demand, **kwargs)


def exponential(n, mean):
    exp = lambda x: e ** (-float(x) / float(mean)) / mean
    return [exp(x) for x in range(0, n)]


def pareto(n, alpha, xm):
    numerator = alpha * xm ** alpha
    par = lambda x: numerator / (x**(alpha+1)) if x >= xm else 0.0
    return [par(x) for x in range(0, n)]


bins = 1101

demand, _ = demandHistogram(bins=bins, density=True)
plt.plot(exponential(bins, 75.77))

xm = amin(demand)
alpha = len(demand) / sum([log(x, e) - log(xm, e) for x in demand])

plt.plot(pareto(bins, alpha, xm))
plt.show()
