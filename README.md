<H3>NAME : SRIRAM S</H3>
<H3>REGISTER NO. : 212222240105</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 07-11-2024 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
Name : R Guruprasad
Reg no. : 212222240033

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io

df=pd.read_csv('Churn_Modelling.csv')
df

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated().sum()

df.drop(['Surname'],axis=1,inplace=True) 
df.drop(['CustomerId','Gender','Geography'],axis=1,inplace=True)
df

df.describe()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

x1=df1.iloc[:,:-1].values
print(x1)
y1=df1.iloc[:,-1].values
print(y1)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

```
## OUTPUT:

### The Dataset:
![image](https://github.com/user-attachments/assets/20d15b52-7c53-4f49-851e-d2cac3981c82)

### Splitting the dataset: 
![image](https://github.com/user-attachments/assets/d7541306-08ca-429c-a03e-2a7fbf824cd8)

### Checking for null values:
![image](https://github.com/user-attachments/assets/892f1695-70e7-4ecb-8f8c-a9ec96ac2a87)

### Checking for duplication:
![image](https://github.com/user-attachments/assets/2075962b-c497-48d8-b405-f072a1e9c86a)

### Dropping unwanted features:
![image](https://github.com/user-attachments/assets/12abc2d9-a0b0-4897-9126-8bdb3dc0432b)

### Describing the dataset:
![image](https://github.com/user-attachments/assets/c79e4e31-71b6-493a-853b-f59b59414d57)

### Scaling the values:
![image](https://github.com/user-attachments/assets/eb9a7660-0d6d-4698-864e-bf90d144e0f6)

### X and Y features:
![image](https://github.com/user-attachments/assets/316c5ced-4649-4861-babb-bdaeebb47543)

### Splitting the training and testing dataset:
![image](https://github.com/user-attachments/assets/5349ba7e-0157-4e43-a195-d472146f9e24)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


