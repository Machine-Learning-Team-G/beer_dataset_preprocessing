import pandas as pd
import sys
from matplotlib import pyplot as plt


# --- 1. read files ---
print("Loading data...")

try:
    df1 = pd.read_csv('beer_reviews.csv', encoding="utf-8") # train data
    df2 = pd.read_csv('train.csv', encoding="utf-8") # test data
    df3 = pd.read_csv('training_set.csv', encoding="utf-8") # train data to preprocess 
    df4 = pd.read_csv('testing_set.csv', encoding="utf-8") # test data to preprocess
except FileNotFoundError:
    print("ERROR: 'dataset' folder not found or files (beer_reviews.csv, train.csv) are missing.")
    print("Please ensure the files are in the correct directory.")
    sys.exit()

print("Data loaded.")


# --- 2. find the number of samples of dataset ---

x1 = ['train_set', 'test_set']
y1 = [len(df1), len(df2)]
x2 = ['preprocessing_training_set', 'preprocessing_testing_set']
y2 = [len(df3), len(df4)]

plt.figure()
plt.title("The number of samples of dataset")
plt.bar(x1,y1)
plt.show()

plt.figure()
plt.title("The number of samples of dataset")
plt.bar(x2,y2)
plt.show()

# --- 3. find correlation between beer style to review_overall

x3 = df3['beer_style_encoded']
y3 = df3['review_overall']

plt.figure()
plt.scatter(x3 ,y3)
plt.show()
