# 支持向量机算法（SVM）

找到一条最适合线来把两类数据分开。

**最适合的线：**离两类数据最远的那条线。例如：两个点的最适合的线就是垂直中分线；  两天不相交的线就是他们的中间的线。

好处：离数据越远，分类错误风险低，分类越稳定。

例：使用逻辑回归来分类，他只是尽可能的用一条线来分割，来让数据分为两类，但是出现了一下异常数据，就会造成分类错误。

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251210203601932.png" alt="image-20251210203601932" style="zoom:33%;" />

加入有一个红点在靠下，就可能导致分线下也向下，会导致错误。

**支持向量机算法：**

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251210203711937.png" alt="image-20251210203711937" style="zoom:33%;" />

是间隔最大（左边界与右边界的垂直距离最大），这样就会减少一下错误数据的感扰。

### 支持向量机算法任务：

找到一组权重参数，使左边界与右边界的间隔距离最大，左边界与右边界中间的线就是**最佳分割线**（超平面）。

生活例子：两个排队区域，一个是买单次票，一个是刷码进站。为了不能让他们搞混，即A可能到B这边，B可能到A这边，就需要一条最佳分割线，使两者的间隔距离最大。

## 如何做到向量机算法：

**决定向量机的不是所有数据点，而是两类数据最边缘的点**（这个就是支持向量），通过两类数据最边缘的点，就能够确定左边界和右边界，**左边界与右边界最中间的那一条线最佳分割线**

## 当两类数据用直线分不开：

有一写两类数据可能混在一起，红球旁边就有蓝球，用普通的线一定无法分割出来，**所以这个时候就用超平面和核函数，利用更高纬度，比如可以用面来分割两类数据，可以用更到纬度来进行分割**

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20251210210026291.png" alt="image-20251210210026291" style="zoom:50%;" />

```
from sklearn import svm
from sklearn.model_selection import train_test_split
import re
import numpy as np


def read(file):
    urls=[]
    with open(file) as f:
        for line in f:
            url=line.strip()
            urls.append(url)
    return urls
def get_f1(url):  #提取长度
    return len(url)

def get_f2(url):    #是否包含http或者https
   f2=re.search(r"(http://)|(https://)",url,re.IGNORECASE)
   if f2:
       return 1
   else:
       return 0

def get_f3(url):       #恶意字符的数量
    f3=re.findall(r"[<>,\'\"/]",url,re.IGNORECASE)
    return len(f3)

def get_f4(url):        #恶意关键词数量
    f4=re.findall(r"(alert)|(script=)|(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)",url,re.IGNORECASE)
    return len(f4)

def get_f5(url):
    f5=re.search(r"/$",url,re.IGNORECASE)
    if f5:
        return 1
    else:
        return 0

def get_features(urls):
    features=[]
    for url in urls:
        f1=get_f1(url)
        f2=get_f2(url)
        f3=get_f3(url)
        f4=get_f4(url)
        f5=get_f5(url)
        features.append([f1,f2,f3,f4,f5])
    return features

#xss数据集
urls=read(r"/SVM/xss_data\xssed.txt")
xss_features=get_features(urls)
xss_labels=len(xss_features)*[1]


#normal数据集
urls=read=(r"D:\python代码\机器学习\SVM\xss_data\dmzo_nomal.txt")
normal_features=get_features(urls)
normal_labels=len(normal_features)*[0]

data=xss_features+normal_features
labels=xss_labels+normal_labels
print(len(data),len(labels))

x_train,x_test,y_train,y_test=train_test_split(data,labels,random_state=2003,train_size=0.8)

model=svm.SVC(kernel='linear',C=1)  #核函数：linear   C为正则化系数

model.fit(x_train,y_train)

predicts=model.predict(x_test)

count=np.count_nonzero(predicts==y_test)

print("准确率：",count/(len(predicts)))
# xss_input=[input()]
# print(xss_input)
# result=get_features(xss_input)
# print(result)
# p=model.predict(result)
# print(p)
```

