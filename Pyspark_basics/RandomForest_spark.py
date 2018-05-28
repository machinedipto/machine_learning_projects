import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.pipeline import Pipeline


spark = SparkSession.builder.master(
    'local').appName('random_forest').getOrCreate()

dogdf = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Tree_Methods/dog_food.csv', inferSchema=True, header=True)

dogdf.show()

dogdf.describe().show()

dogdf.groupby('spoiled').count().show()

dogdf.columns

assembler = VectorAssembler(
    inputCols=['A', 'B', 'C', 'D'], outputCol='features')

output = assembler.transform(dogdf)

output.show()

final_data = output.select('features', 'Spoiled')

final_data = final_data.withColumnRenamed('Spoiled', 'label')

final_data.show()

final_data.show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])

rf = RandomForestClassifier()

rf.labelCol
rf.params

paramgrid = ParamGridBuilder().addGrid(rf.maxDepth, [1, 2, 3, 4]).addGrid(rf.minInstancesPerNode, [
    1, 3, 5, 10, 50, 100]).addGrid(rf.numTrees, [20, 40, 80, 100, 200]).build()

crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramgrid,
                          evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvModel = crossval.fit(train_data)


cvModel.bestModel._java_obj.getMaxDepth()

cvModel.bestModel.featureImportances
cvModel.

prediction = cvModel.transform(test_data)

prediction.show()
