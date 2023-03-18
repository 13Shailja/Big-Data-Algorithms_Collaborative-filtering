''' Rating Prediction using Linear Regression '''
# Importing libraries
import time
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when
from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName('LR').getOrCreate()

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
def vector_assembler(dataframe, indep_cols):    
    assembler = VectorAssembler(inputCols = indep_cols,
                                outputCol = 'features')
    output = assembler.transform(dataframe).drop(*indep_cols)    
    return output

df = vector_assembler(ratings, indep_cols = ratings.drop('rating').columns)
testdf = vector_assembler(testRatings1, indep_cols = testRatings1.columns)


# Training the Linear Regression model
start_train_time = time.time()
lr = LinearRegression(labelCol = 'rating',
                      featuresCol = 'features',
                      regParam = 0.3) #avoid overfitting

lr = lr.fit(df)
end_train_time = time.time()

# Making prediction
start_pred_time = time.time()
pred = lr.transform(testdf)
end_pred_time = time.time()

prediction_col = round(pred.select("prediction").toPandas())
prediction_col.to_csv (r'LR_Predictions.dat', index = None, header=False) 

# Evaluating our model
evaluation_file=open('linear_regression_evaluation.txt', 'w')
evaluation_file.write(f'Evaluation of the model: \n')
evaluation_file.write(f'Time taken for training the model : {end_train_time - start_train_time} seconds\n')
evaluation_file.write(f'Time taken for making predictions : {end_pred_time - start_pred_time} seconds\n')

# Print the coefficients and intercept for linear regression
evaluation_file.write("Coefficients: %s" % str(lr.coefficients))
evaluation_file.write("Intercept: %s" % str(lr.intercept))


# Summarize the model over the training set and print out some metrics
trainingSummary = lr.summary
evaluation_file.write("numIterations: %d" % trainingSummary.totalIterations)
evaluation_file.write("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
evaluation_file.write("RMSE: %f" % trainingSummary.rootMeanSquaredError)
evaluation_file.write("r2: %f" % trainingSummary.r2)

evaluation_file.close()

spark.stop()