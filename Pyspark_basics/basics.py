import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField, IntegerType

spark = SparkSession.builder.master(
    'spark://Light-Ubuntu:7077').appName('Basics').getOrCreate()

df = spark.read.json('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/people.json')

df.show()

df.printSchema()

data_schema = [StructField('age', IntegerType(), True),
               StructField('name', StringType(), True)]


final_struc = StructType(fields=data_schema)

# This is how you create a schema and while reading you can pass it as inferredschema

df.columns

type(df['age'])

df.select('age').show()

# this two has a diference which is first one is returning the type column
# the second select is returning a dataframe

df.head(2)[0]

df.withColumn('new_column', 1)

# create a temp view to look on the
df.createOrReplaceTempView('people')

results = spark.sql("select * from people where age = 30")

results.show()


spark.stop()
