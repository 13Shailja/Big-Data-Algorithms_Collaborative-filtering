''' Rating Prediction using Linear Regression '''
# Importing libraries
import time
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel


spark = SparkSession.builder.appName('RandomForest').getOrCreate()

# Loading the training dataset
lines = spark.read.text('train.dat').rdd
parts = lines.map(lambda row: row.value.split("\t"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1]), rating=int(p[2])))

ratings = spark.createDataFrame(ratingsRDD)
ratings = ratings.select("*").withColumn("id", monotonically_increasing_id())
(training, test) = ratings.randomSplit([0.8, 0.2])
# ratings.count() # 85724

# Loading the Test dataset
lines = spark.read.text('test.dat').rdd
parts = lines.map(lambda row: row.value.split("\t"))
testRatingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1])))

testRatings = spark.createDataFrame(testRatingsRDD)
testRatings = testRatings.select("*").withColumn("id", monotonically_increasing_id())
# testRatings.count() # 2154

# Data Cleaning

# item_replacement_for_user819 contains a single row, chosen at random from the 90% of ratings data for user "819".
item_replacement_for_user819 = ratings.filter(ratings.userId == "819").sample(False, 0.9).limit(1)
# Extracting the value of the itemId column from item_replacement_for_user819 DataFrame, and assigning it to a new variable called iId_819.
iId_819 = item_replacement_for_user819.first()['itemId']

# Repeating the extraction process to extract two ItemId's for user "218"
item_replacement_for_user218 = ratings.filter(ratings.userId == "218").sample(False, 0.9).limit(2)
iId_218_1 = item_replacement_for_user218.first()['itemId']
iId_218_2 = item_replacement_for_user218.select('itemId').collect()[1][0]

# Replacing the found Item Id above so we have valid items in our dataset
testRatings1 = testRatings.withColumn("itemId", when(testRatings.itemId == "375",iId_819)
                                 .when(testRatings.itemId == "35",iId_218_1)
                                 .when(testRatings.itemId == "30" ,iId_218_2)
                                 .otherwise(testRatings.itemId))

# Matched the user and item so there are no missing information for our matrix
missing_user_inTraining = testRatings1.select('userId').subtract(ratings.select('userId')).count()
missing_item_inTraining = testRatings1.select('itemId').subtract(ratings.select('itemId')).count()
missing_user_inTraining, missing_item_inTraining

# Preparing the data

# Using Vector Assembler
# VectorAssembler is a feature transformer in Apache Spark that combines multiple columns of data into a single vector column. 
# It is commonly used for preparing data for machine learning models that require input features in vector format.
assembler = VectorAssembler(inputCols=["userId", "itemId"], outputCol="features")

trainDF = assembler.transform(ratings)
testDF = assembler.transform(testRatings)
(training, test) = trainDF.randomSplit([0.8, 0.2], seed=42)
(training, valid) = trainDF.randomSplit([0.9, 0.1], seed=42)

# Training the Random Forest model
# RandomForestRegressor can predict numerical values based on input features. 
# It works by creating an ensemble of decision trees that collectively make predictions on the input data. 
# Each decision tree is trained on a random subset of the features and data, 
# and the final prediction is calculated as the average of the predictions from all the trees in the forest. 
start_train_time = time.time()
rf = RandomForestRegressor(featuresCol='features', labelCol='rating', 
                           numTrees=10, maxMemoryInMB=1024,
                           subsamplingRate=0.3)
rf_model = rf.fit(training)
end_train_time = time.time()

# Making prediction
start_pred_time = time.time()
test_set = rf_model.transform(testDF)
end_pred_time = time.time()

prediction_col = round(test_set.select("prediction").toPandas())
prediction_col.to_csv (r'random_forest.dat', index = None, header=False) 

# Evaluating our model
evaluation_file=open('random_forest_evaluation.txt', 'w')
evaluation_file.write(f'Evaluation of the model: \n')
evaluation_file.write(f'Time taken for training the model : {end_train_time - start_train_time} seconds\n')
evaluation_file.write(f'Time taken for making predictions : {end_pred_time - start_pred_time} seconds\n')

# Even though there are only 2 features, saving their feature Importances 
evaluation_file.write(str(rf_model.featureImportances))
evaluation_file.close()

spark.stop()