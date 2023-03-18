# Import libraries
import pandas as pd
import time
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split

# Predict function
def predict_ratings(df):
  rating_file=open('svd_surprise_prediction.dat', 'w')
  
  for ind in df.index:
    userId = df['userId'][ind]
    itemId = df['itemId'][ind]
    pred = round(model.predict(userId, itemId).est)
    rating_file.write(str(pred))
    if ind < len(df.index) - 1:
      rating_file.write('\n')
  
  rating_file.close()

# Reading training file
trainDF = pd.read_csv('train.dat', sep='\t', names=["userId", "itemId", "rating", "timestamp"])
trainDF.drop('timestamp', axis=1, inplace=True)
# Reading test file
testDF = pd.read_csv('test.dat', sep='\t', names=["userId", "itemId"])

# Cleaning the Test file
iId_819 = int(trainDF.query("userId == 819").sample(n=1)["itemId"])
iId_218_1 = int(trainDF.query("userId == 218").sample(n=1)["itemId"])
iId_218_2 = int(trainDF.query("userId == 218").sample(n=1)["itemId"])

testDF['itemId'] = testDF['itemId'].replace([375], iId_819)
testDF['itemId'] = testDF['itemId'].replace([30], iId_218_1)
testDF['itemId'] = testDF['itemId'].replace([35], iId_218_1)

# Using surprise for SVD
# Data Preparation
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(trainDF, reader)
trainset = data.build_full_trainset()

# Training the model
# Using SVD.
# Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three matrices, 
# where the middle matrix contains the singular values. 
start_train_time = time.time()
model = SVD(n_factors=50)
model.fit(trainset)
end_train_time = time.time()

start_pred_time = time.time()
predict_ratings(testDF)
end_pred_time = time.time()

# Evaluating our model
evaluation_file=open('svd_surprise_evaluation.txt', 'w')
evaluation_file.write(f'Time taken for training the model : {end_train_time - start_train_time} seconds\n')
evaluation_file.write(f'Time taken for making predictions : {end_pred_time - start_pred_time} seconds\n')

# Run 5-fold cross-validation and then print results
eval = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
evaluation_file.write(f'Evaluation of the model: \n')
evaluation_file.write(f'{eval}')
evaluation_file.close()