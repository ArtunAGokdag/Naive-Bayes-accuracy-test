# %%
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# Load data from file
test_df = pd.read_csv('testSet.csv', na_values=['?'])
train_df = pd.read_csv('trainSet.csv', na_values=['?'])


# Preprocessing The Data


# Dummy encoding property_magnitude
# For train set
ohe = OneHotEncoder(sparse=False)
encoded_array = ohe.fit_transform(train_df[['property_magnitude']])

labels = np.array(ohe.categories_).ravel()
encoded_df = pd.DataFrame(encoded_array, columns=labels)
train_df = pd.concat([train_df, encoded_df], axis=1)

# For test set
ohe = OneHotEncoder(sparse=False)
encoded_array = ohe.fit_transform(test_df[['property_magnitude']])

labels = np.array(ohe.categories_).ravel()
encoded_df = pd.DataFrame(encoded_array, columns=labels)
test_df = pd.concat([test_df, encoded_df], axis=1)


# Map employment
# For train
train_df['employment_mapped'] = train_df['employment'].map({
    '>=7' : 4,
    '4<=X<7' : 3,
    '1<=X<4': 2,
    '<1' : 1,
    'unemployed' : 0
})

#for test
test_df['employment_mapped'] = train_df['employment'].map({
    '>=7' : 4,
    '4<=X<7' : 3,
    '1<=X<4': 2,
    '<1' : 1,
    'unemployed' : 0
})



# Map Credit History
# For train
train_df['credit_history_mapped'] = train_df['credit_history'].map({
    "'all paid'": 5,
    "'existing paid'": 4,
    "'no credits/all paid'": 3,
    "'delayed previously'": 2,
    "'critical/other existing credit'": 1
})

#for test
test_df['credit_history_mapped'] = test_df['credit_history'].map({
    "'all paid'": 5,
    "'existing paid'": 4,
    "'no credits/all paid'": 3,
    "'delayed previously'": 2,
    "'critical/other existing credit'": 1
})

# Remove Mising values

train_df = train_df.dropna()
test_df = test_df.dropna()


#Split the dataset dependent and independet variables
x_train = train_df[["'life insurance'",
                    "'no known property'", 
                    "'real estate'", 'car', 
                    'employment_mapped', 
                    'credit_history_mapped',
                    'credit_amount',
                    'age']]

x_test = test_df[["'life insurance'",
                    "'no known property'", 
                    "'real estate'", 'car', 
                    'employment_mapped', 
                    'credit_history_mapped',
                    'credit_amount',
                    'age']]

y_train = train_df.loc[:, 'class']
y_test = test_df.loc[:, 'class']

# Normalize values
# For train
scaler = MinMaxScaler()
scaler.fit(x_train)
scaled = scaler.fit_transform(x_train)
x_train_norm = pd.DataFrame(scaled, columns=x_train.columns)

# for test
scaler = MinMaxScaler()
scaler.fit(x_test)
scaled = scaler.fit_transform(x_test)
x_test_norm = pd.DataFrame(scaled, columns=x_test.columns)


# Turn string values in class variable to integer
y_train = y_train.map({'good' : 1, 'bad': 0})
y_test = y_test.map({'good' : 1, 'bad': 0})


# #### Train Model
# Gaussian Naive Bayes from sci-kit learn library

model = GaussianNB()
model.fit(x_train_norm, y_train)

# get prediction values in y_pred
y_pred = model.predict(x_test_norm)
y_true = np.array(y_test) # actual values

### RUN TESTS

def accuracy(y_true, y_pred):
	accuracy = np.sum(y_true == y_pred) / len(y_true)
	return accuracy

acc = accuracy(y_true, y_pred)

#TRUE POSITIVE
# Where both are 1
TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))

# TRUE NEGATIVE
# Where both are 0
TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))


# FALSE POSITIVE
# where prediction is 1 and ture is 0
FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))



# FALSE NEGATIVE
# Where prediction is 0 and true is 1
FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))



# Alternative way to calculate accuracy
acc_alt = (TN + TP) / len(y_pred)


# TRUE POSITIVE RATE
TP_rate = TP / (TP + FN)


# TRUE NEGATIVE RATE
TN_rate = TN / (TN + FP)

# Used np.logical_and as a way of using 2 conditions in np.sum

# PRINT
print("Test results:")
print(f"{'Accuracy:':20s}{acc:.3f}")
print(f"{'TP rate:':20s}{TP_rate:.3f}")
print(f"{'TN rate:':20s}{TN_rate:.3f}")
print(f"{'TP count:':20s}{TP}")
print(f"{'TN count:':20s}{TN}")