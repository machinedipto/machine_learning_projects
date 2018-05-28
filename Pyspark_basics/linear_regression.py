import findspark
findspark.init("/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/")

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import corr

spark = SparkSession.builder.master(
    "local").appName("linear_reg").getOrCreate()

shipdf = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Linear_Regression/cruise_ship_info.csv', inferSchema=True, header=True)

shipdf.show()
shipdf.describe().show()

shipdf.groupby('Cruise_line').count().show()

# converting the categorical data to label indexing

indexer = StringIndexer(inputCol='Cruise_line', outputCol='Cruise_cat')
indexed = indexer.fit(shipdf).transform(shipdf)

indexed.show()

# Creating vector VectorAssembler

indexed.columns

assembler = VectorAssembler(inputCols=['Age',
                                       'Tonnage',
                                       'passengers',
                                       'length',
                                       'cabins',
                                       'passenger_density',
                                       'Cruise_cat'], outputCol='features')

output = assembler.transform(indexed)

output.show()

output.select('features', 'crew').show()

final_data = output.select('features', 'crew')

final_data.describe().show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()
lr = LinearRegression(labelCol='crew')

lrmodel = lr.fit(train_data)

print("Coefficients {} Intercept{}".format(
    lrmodel.coefficients, lrmodel.intercept))

test_results = lrmodel.evaluate(test_data)

print("RMSE{}".format(test_results.rootMeanSquaredError))
print("R2{}".format(test_results.r2))


shipdf.select(corr('crew', 'passengers')).show()

spark.stop()
