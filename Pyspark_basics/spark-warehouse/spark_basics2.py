import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.1.1-bin-hadoop2.6/')
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, avg, stddev, mean, year, dayofyear, month, hour, weekofyear, date_format

spark = SparkSession.builder.master(
    'spark://Light-Ubuntu:7077').appName('OPS').getOrCreate()

df = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/appl_stock.csv', inferSchema=True, header=True)


df.printSchema()
df.show()

df.filter((df['close'] < 200) & (df['open'] > 200)).show()

result = df.filter(df['Low'] == 197.16).collect()
row = result[0]

row.asDict()['Volume']


df = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/sales_info.csv', inferSchema=True, header=True)


df.show()
df.printSchema()
df.groupBy("Company").mean().show()

    group_data = df.groupby("Company")

    group_data.agg({'Sales': 'max'}).show()


df.select(avg('Sales').alias('avg')).show()


df = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/ContainsNull.csv', inferSchema=True, header=True)

df.show()

# filling miisng value with 0 in sales columns

df.na.fill(0, subset=['Sales']).show()

# filling miising value with mean

df.na.fill(df.select(avg(df['Sales'])).collect()
           [0][0], subset=['Sales']).show()


# example on timestamp

df = spark.read.csv('/media/light/208C3A738C3A4396/Accesories/Udemy - Spark and Python for Big Data with PySpark/01 Introduction to Course/attached_files/002 Course Overview/Python-and-Spark-for-Big-Data-master/Spark_DataFrames/appl_stock.csv', inferSchema=True, header=True)


df.show()

# create a new column with year

newdf = df.withColumn("year", year(df['Date']))

newdf.groupBy("year").mean().show()
