<H3>ENTER YOUR NAME : Sriram S</H3>
<H3>ENTER YOUR REGISTER NO. 212222240105</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22.08.2024</H3>
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
```Python
import pandas as pd                                                 # Importing Libraries
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         # Read the dataset from drive
df.head()
```
```Python
df.isnull().sum()                                                   # Finding Missing Values
```
```Python                                               
df.duplicated().sum()                                               # Check For Duplicates
```
```Python                                              
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               # Remove Unnecessary Columns
scaler=StandardScaler()                                             # Normalize the dataset
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
```
```Python
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     # Split the dataset into input and output
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   # Splitting the data for training & Testing
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     # X Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   # Y Train and Test
```


## OUTPUT:
### DATASET:
![image](https://github.com/user-attachments/assets/4e75215a-6909-47e8-be89-f1b0dcf64069)

### NULL VALUES: 
![image](https://github.com/user-attachments/assets/1a4ab591-3115-42e5-88c6-c2356d1175b5)

### NORMALIZED DATA:
![image](https://github.com/user-attachments/assets/27a5d162-c488-42b5-b5b9-fe563cf3062c)
### DATA SPLITTING:
![image](https://github.com/user-attachments/assets/e5154e4d-b3f4-444e-9d20-5a3d908c4797)
![image](https://github.com/user-attachments/assets/6d65b734-6632-4fa7-aa91-39c6feb88e5e)

### TRAIN AND TEST DATA:
![image](https://github.com/user-attachments/assets/8716f8a6-4f08-42ea-9c6a-1b86bb38ae00)
![image](https://github.com/user-attachments/assets/4214bcfc-f2c1-4f69-b7b0-6ead12703483)
![image](https://github.com/user-attachments/assets/f219a314-14c7-4921-a631-df59b3c46301)
![image](https://github.com/user-attachments/assets/025abc61-60a1-4f7d-b8ca-40210b03f7f4)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python using a data set downloaded from Kaggle.


