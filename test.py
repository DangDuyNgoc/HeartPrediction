import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import colorama
from colorama import Back
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu từ file CSV
Main_Dataset = pd.read_csv('heart.csv')

# Hiển thị kích thước của DataFrame
print(Main_Dataset.shape)

print(Main_Dataset.head(10))
print(Main_Dataset.info())
print(Main_Dataset.isnull().sum())
print(Main_Dataset.describe())

# Overview data
def describe(Main_Dataset):
    
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    min_ = []
    max_ = []
    
    for item in Main_Dataset.columns:
        variables.append(item)
        dtypes.append(Main_Dataset[item].dtype)
        count.append(len(Main_Dataset[item]))
        unique.append(len(Main_Dataset[item].unique()))
        missing.append(Main_Dataset[item].isna().sum())
        
        if Main_Dataset[item].dtypes == 'float64' or Main_Dataset[item].dtypes == 'int64':
            min_.append(Main_Dataset[item].min())
            max_.append(Main_Dataset[item].max())
        else:
            min_.append('Str')
            max_.append('Str')
        

    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing,
        'Min': min_,
        'Max': max_
    })    
        
    return output

desc_df = describe(Main_Dataset)
print(desc_df)

# Define the continuous features
Numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig = plt.figure(figsize=[15, 10], dpi=100)
for i in range(len(Numerical)):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x=Numerical[i], data=Main_Dataset, boxprops=dict(facecolor="#E72B3B"), patch_artist=True)

# plt.tight_layout()
# plt.show()

Chol_noise = Main_Dataset[Main_Dataset["chol"]>400].index
print(f"Chol_noise: ", Chol_noise)

Main_Dataset.drop(index=Chol_noise, inplace=True)
Main_Dataset.shape


fig = plt.figure(figsize = [15,3], dpi=200)
sns.boxplot(x = 'chol', data = Main_Dataset,
        boxprops = dict(facecolor = "#E72B3B"))
    
# plt.show()
plt.close('all')

Target_0_data = Main_Dataset[Main_Dataset["target"]==0]
Target_0_data = pd.DataFrame(Target_0_data)
Target_1_data = Main_Dataset[Main_Dataset["target"]==1]
Target_1_data = pd.DataFrame(Target_1_data)
print("The shape of data when target is '0': Not disease",Target_0_data.shape)
print("The shape of data when target is '1': Disease",Target_1_data.shape)

Target_0_data.sort_values(by=['age'], inplace=True)
Target_1_data.sort_values(by=['age'], inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=100)
sns.barplot(x= Target_0_data['age'], y= Target_0_data['cp'], errorbar=None,
            palette="dark:salmon_r",ax= axes[0]).set(title='Age - CP in Heart Disease = 0')
sns.barplot(x= Target_1_data['age'], y= Target_1_data['cp'], errorbar=None,
            palette="dark:salmon_r",ax= axes[1]).set(title='Age - CP in Heart Disease = 1')

plt.tight_layout()
plt.show()

Features = Main_Dataset.drop(columns='target')
Features = pd.DataFrame(Features)
scaler = MinMaxScaler()
Norm_data = scaler.fit_transform(Features)
Norm_df = pd.DataFrame(Norm_data, columns= Features.columns)

desc_norm_df = describe(Norm_df)
print(desc_norm_df)

print(Norm_df.head(10))