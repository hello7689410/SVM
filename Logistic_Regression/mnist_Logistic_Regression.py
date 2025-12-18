from keras.datasets import mnist
from sklearn import linear_model
import numpy as np


#得到训练数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#对训练数据展平
x_train=x_train.reshape(x_train.shape[0],-1).astype('float')/255
x_test=x_test.reshape(x_test.shape[0],-1).astype('float')/255

model=linear_model.LogisticRegression()
model.fit(x_train,y_train)
predicts=model.predict(x_test)
corrects=np.count_nonzero(predicts==y_test)
print("准确度：",corrects/len(x_test))
