import pandas as pd
import numpy as np
import os

# Import the dataset
dataset = pd.read_csv(
    str(os.getcwd())+'/data/ad.data',
    header=None,
    encoding='utf-8',
    low_memory=False)

# Handle missing data
dataset = dataset.replace('?',np.nan)
dataset = dataset.replace(r'\s*\?\s*', np.nan, regex=True)

# Replace null values with mode value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset.iloc[:,0:4] = imputer.fit_transform(dataset.iloc[:,0:4])

# Label encode the y attribute
# 0 - ad.
# 1 - nonad.
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset.iloc[:,1558] = label_encoder.fit_transform(dataset.iloc[:,1558])

# Describe the dataset
dataset_describe = dataset.describe()

# Feature extraction
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.astype(int)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

n = 5

model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model,n_features_to_select=n)
fit = rfe.fit(X,y)
number_of_features = fit.n_features_
selected_features = fit.support_
feature_ranking = fit.ranking_

def rank_features(feature_ranking):
    ones = np.where(feature_ranking == 1)[0]
    return ones

print(rank_features(feature_ranking))