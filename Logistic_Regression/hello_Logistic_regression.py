from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
import numpy as np


iris=datasets.load_iris()
target=iris.target
data=iris.data
data=data[:,:2] #
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=25)
model=linear_model.LogisticRegression()
model.fit(x_train,y_train)
predicts=model.predict(x_test)
corrects=np.count_nonzero(predicts==y_test)
print("准确率：",corrects/len(y_test))
