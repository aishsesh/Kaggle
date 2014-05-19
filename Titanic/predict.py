import numpy as np
import pandas as pd

def dataprocessing(df):

	df = df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Embarked'], axis=1) # dropping some features
	median = df.Age.median() # filling missing age values with the median age (age was right skewed)
	df.loc[df.Age.isnull(), 'Age'] = median
	df['Minor'] = df['Age'].map( lambda x: 1*(x < 18) )
	#Parch is 1 if passenger has a parent and child on board. 
	#If the passenger has a child on board there was a higher likelihood of them surviving as they could accompany the child on the lifeboat
	#Assuming that people between ages 18 and 40 could be accompanying parents	
	df['AccompanyingParent'] = (~df.Minor & df.Parch)	
	df['AccompanyingParent'] = df.AccompanyingParent & (df.Age < 40)
	df = df.drop(['Age','Parch' ], axis=1)
	df['Gender'] = df.Sex.map({'female': 0, 'male': 1}).astype(int)
	df['AccompanyingParent'] = df.AccompanyingParent.astype(int)
	df = df.drop(['Sex'], axis=1)
	return df


train = pd.read_csv ('train.csv', header =0)
train = dataprocessing(train)
train_data = train.values


test = pd.read_csv ('test.csv', header =0)
test = dataprocessing(test)
test_data = test.values

#check no NaN values =- shud return empty arrays
test[test.Fare.isnull()] 
test.loc[test.Fare.isnull(), 'Fare'] = test.Fare.median()
test_data = test.values



from sklearn.linear_model import LogisticRegression
l = LogisticRegression()

l.fit(train_data[0::,2::], train_data[0::,1])
prediction = l.predict(test_data[0::,1::])

np.savetxt("prediction.csv", prediction, delimiter = ",")