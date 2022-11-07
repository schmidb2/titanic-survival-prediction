import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#read in data
data = pd.read_csv('passenger_list.csv')

#drop irrelevant data
data= data.drop(columns=['home.dest','body','boat','cabin','name','ticket'],axis=1)

#replace features that are strings with equivalents int
data.replace({'sex':{'male':0,'female':1},'embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

#replace nan values with mean
data['age'].fillna(data['age'].mean(),inplace=True)
data['fare'].fillna(data['fare'].mean(),inplace=True)
data['embarked'].fillna(data['embarked'].mean(),inplace=True)

#remove label from data into separate variable 
label = data['survived']
train = data.drop(columns = ['survived'])

#split data into train and test (20% holdout)
x_train,x_test,y_train,y_test = train_test_split(train,label, test_size=0.2,random_state=2)

#creating model
model = LogisticRegression()
model.fit(x_train,y_train)

#use model to get predictions
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test,predictions)
print("Accuracy = ",accuracy)

#create confusion matrix
matrix = confusion_matrix(y_test,predictions)
#plot confusion matrix
plt.figure(figsize = (10,7)) #plotting confusion matrix using seaborn
sn.heatmap(matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth') 
plt.show()

    



