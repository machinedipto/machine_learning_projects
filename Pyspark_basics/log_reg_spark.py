import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')

import pandas
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.master('local').appName('log_reg').getOrCreate()

spark.stop()

customerdf = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Logistic_Regression/customer_churn.csv', inferSchema=True, header=True)

customerdf.show()

customerdf.printSchema()

customerdf.describe().show()

customerdf.columns

customerdf = customerdf.drop('Account_Manager')


customerdf.groupBy('Churn').count().show()
customerdf = customerdf.drop('Names')

customerdf.dtypes
cat_columns = [item[0] for item in customerdf.dtypes if item[1] == 'string']


int_columns = [item[0] for item in customerdf.dtypes if (
    item[1] == 'double' or item[1] == 'int')]

int_columns

int_columns.remove('Churn')

int_columns

customer_cat = customerdf.select(cat_columns)

customer_cat.show()

customerdf.show()

cat_columns

string_indexer = [
    StringIndexer(inputCol=x, outputCol='idx_{}'.format(x))
    for x in cat_columns
]

# customerdf.show()
#
# string_indexer[0]
#
# indexer = string_indexer.fit(customerdf)
# indexed = indexer.transform(customerdf)


# indexed.columns

encoder = [
    OneHotEncoder(inputCol='idx_{0}'.format(x), outputCol='enc_{0}'.format(x))
    for x in cat_columns
]

int_columns

inputcols = ['idx_{0}'.format(x) for x in cat_columns] + int_columns

inputcols


print(inputcols)

assembler = VectorAssembler(inputCols=inputcols, outputCol='features')

train_data, test_data = customerdf.randomSplit([0.7, 0.3])

log_reg = LogisticRegression(featuresCol="features", labelCol="Churn")

pipeline = Pipeline(stages=string_indexer + encoder + [assembler])

train_prepared = pipeline.fit(customerdf).transform(customerdf)

train_enocded.show()

train_enocded.show()

train_prepared.select('features').head(1)

final_train = train_prepared.select('features', 'Churn')

final_train.show()

# calling new customer
new_cust = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_for_Machine_Learning/Logistic_Regression/new_customers.csv', inferSchema=True, header=True)

test_preapred = pipeline.fit(new_cust).transform(new_cust)


test_prepared.select('features').head(1)

final_test = test_preapred.select('features')


log_model = log_reg.fit(final_train)

final_pred = log_model.transform(test_prepared)

final_pred.show()

final_pred.select('pr').show()

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                        labelCol='Churn')

final_pred.select('Churn').show()


spark.stop()


#############
