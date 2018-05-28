import sklearn
from numpy import array
from numpy import argmax
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#load data
data = pd.read_csv("flavors_of_cacao.csv", delimiter = ",", encoding = "UTF-8")
print("Column names: ", data.columns.values)
print("Dataframe shape: ", data.shape)

# print bean type data to check null values
# print(data['BeanType'].tolist())

#count missing values
print("\nMissing values statistics:")
missing_values_count = data.isnull().sum()
print(missing_values_count)

# print data types
print("\nData types:\n", data.dtypes)

# convert percent column to float
data['CocoaPercent'] = data['CocoaPercent'].apply(lambda x: float(x.strip('%'))/100)
#print("percent column: ", data['CocoaPercent'])

# select columns to rescale
int_features = data.select_dtypes(include=['int64']).copy()
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(int_features)
df_rescaled = pd.DataFrame(rescaled_features)
df_rescaled.columns = ['ref', 'date']
data = pd.concat([data, df_rescaled], axis=1)
# summarize transformed data
np.set_printoptions(precision=3)
print("rescaled int features: ")
print(rescaled_features)

# select object columns to one-hot encode
obj_features = data.select_dtypes(include=['object']).copy()
print("object columns: ", obj_features.columns)

# # encode values
# company_values = array(data['Company(Maker-if known)'])
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(company_values)
#
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#
# # # invert first example
# # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# # print(inverted)
#
# #df_companies = pd.get_dummies(data['Company(Maker-if known)'])
# #print(df_companies)

for columnName in obj_features.columns.values:
    df_new = pd.get_dummies(data[columnName])
    data = pd.concat([data, df_new], axis=1)
print(data.head())

# drop initial, unprocessed columns
new_data = data.drop(columns = ['Company(Maker-if known)', 'SpecificBeanOrigin or BarName', 'REF', 'ReviewDate', 'CompanyLocation', 'BeanType', 'BroadBeanOrigin'])
# shuffle data
new_data = shuffle(new_data)
print("new data\n", new_data.head())

# separate data into features and target
target = new_data['Rating']
features = new_data.loc[:, new_data.columns != 'Rating']
# print("\nFeatures: ", features.columns)

size = new_data.shape[0]
# select first 60% as train data
train = new_data[:int(0.6 * size)]
#print("\nTrain size: ", train.shape)

# select next 20 % as cross validation
cross = new_data[int(0.6 * size):int(0.8*size)]
#print("Cross validation size: ", cross.shape)

# select last 20% as test data
test = new_data[int(0.8*size):]
#print("Test size: ", test.shape)

