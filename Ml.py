# Prediction algorithm using ML

import pandas
from sklearn.tree import DecisionTreeClassifier

mobiledata=pandas.read_csv('mobile.csv')
#print(mobiledata)

#training data
features=mobiledata.drop(columns=['Mobile'])
labels=mobiledata['Mobile']

#build a model
model=DecisionTreeClassifier()
model.fit(features,labels)

#test data and predict
result=model.predict([[68,1,60000],[20,1,5000]])
print(result)