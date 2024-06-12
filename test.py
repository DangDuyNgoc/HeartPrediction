import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from colorama import Back
from sklearn.metrics import confusion_matrix
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

df_cleaned = Main_Dataset.copy()
# Define the continuous features
Numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig = plt.figure(figsize=[15, 10], dpi=100)
for i in range(len(Numerical)):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x=Numerical[i], data=Main_Dataset, boxprops=dict(facecolor="#E72B3B"), patch_artist=True)
        
# plt.tight_layout()
# plt.show()

# Function to remove outliers using IQR method
def remove_outliers(df, numerical_features):
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtering the dataframe
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

# Remove outliers from the dataset
df_cleaned = remove_outliers(df_cleaned, Numerical)

# Plotting box plots before and after removing outliers
fig = plt.figure(figsize=[15, 10], dpi=100)
for i in range(len(Numerical)):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x=Numerical[i], data=df_cleaned, boxprops=dict(facecolor="#E72B3B"), patch_artist=True)
    plt.title(f'After - {Numerical[i]}')

# plt.tight_layout()
# plt.show()


Chol_noise = Main_Dataset[Main_Dataset["chol"]>400].index
print(f"Chol_noise: ", Chol_noise)

Main_Dataset.drop(index=Chol_noise, inplace=True)
print(Main_Dataset.shape)

# fig = plt.figure(figsize = [15,3], dpi=200)
# sns.boxplot(x = 'chol', data = Main_Dataset,
#         boxprops = dict(facecolor = "#E72B3B"))
    
# plt.close('all')
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

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=100)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['cp'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - CP in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['cp'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - CP in Heart Disease = 1')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['fbs'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - fbs in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['fbs'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - fbs in Heart Disease = 1')


# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['restecg'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - restecg in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['restecg'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - restecg in Heart Disease = 1')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['exang'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - exang in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['exang'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - exang in Heart Disease = 1')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['slope'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - slope in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['slope'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - slope in Heart Disease = 1')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['ca'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - ca in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['ca'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - ca in Heart Disease = 1')

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
# sns.barplot(x= Target_0_data['age'], y= Target_0_data['thal'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[0]).set(title='Age - thal in Heart Disease = 0')
# sns.barplot(x= Target_1_data['age'], y= Target_1_data['thal'], errorbar=None,
#             palette="dark:salmon_r",ax= axes[1]).set(title='Age - thal in Heart Disease = 1')

# plt.figure(figsize=(20,5), dpi=200)
# plt.plot(Target_0_data['age'], Target_0_data['trestbps'], color= '#E72B3B')
# plt.scatter(Target_1_data['age'], Target_1_data['trestbps'], color= 'black')

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.legend(["Without Disease","With Disease"], fontsize=15)

# plt.title("Age - Resting blood pressure (in mm Hg)", fontsize=20)
# plt.xlabel("Age", fontsize=15)
# plt.ylabel("trtbps", fontsize=15)

# plt.figure(figsize=(20,5), dpi=200)
# plt.plot(Target_0_data['age'], Target_0_data['chol'], color= '#E72B3B')
# plt.scatter(Target_1_data['age'], Target_1_data['chol'], color= 'black')

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.legend(["Without Disease","With Disease"], fontsize=15)

# plt.title("Age - Cholestoral in mg/dl fetched", fontsize=20)
# plt.xlabel("Age", fontsize=15)
# plt.ylabel("chol", fontsize=15)

# plt.figure(figsize=(20,5), dpi=200)
# plt.plot(Target_0_data['age'], Target_0_data['thalach'], color= '#E72B3B')
# plt.scatter(Target_1_data['age'], Target_1_data['thalach'], color= 'black')

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.legend(["Without Disease","With Disease"], fontsize=15)

# plt.title("Age - thalach", fontsize=20)
# plt.xlabel("Age", fontsize=15)
# plt.ylabel("thalach", fontsize=15)

plt.figure(figsize=(20,5), dpi=200)
plt.plot(Target_0_data['age'], Target_0_data['oldpeak'], color= '#E72B3B')
plt.scatter(Target_1_data['age'], Target_1_data['oldpeak'], color= 'black')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(["Without Disease","With Disease"], fontsize=15)

plt.title("Age - ST depression caused by activity", fontsize=20)
plt.xlabel("Age", fontsize=15)
plt.ylabel("oldpeak", fontsize=15)

# plt.close('all')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(18,15))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])
ax6 = fig.add_subplot(gs[2,0])
ax7 = fig.add_subplot(gs[2,1])
ax8 = fig.add_subplot(gs[2,2])

background_color = "white"
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 
ax3.set_facecolor(background_color) 
ax4.set_facecolor(background_color) 
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 
ax7.set_facecolor(background_color) 
ax8.set_facecolor(background_color) 

# Title of the plot
ax0.spines["bottom"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.text(0.5,0.5,
         'Count plot for various\n categorical features',
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=18, fontweight='bold',
         fontfamily='serif',
         color="#000000")

# Sex count
ax1.text(0.3, 220, 'Sex', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1,data=Main_Dataset,x='sex',palette="dark:salmon_r")
ax1.set_xlabel("")
ax1.set_ylabel("")

# Exang count
ax2.text(0.3, 220, 'Exang', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax2,data=Main_Dataset,x='exang',palette="dark:salmon_r")
ax2.set_xlabel("")
ax2.set_ylabel("")

# Caa count
ax3.text(1.5, 200, 'Caa', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax3,data=Main_Dataset,x='ca',palette="dark:salmon_r")
ax3.set_xlabel("")
ax3.set_ylabel("")

# Cp count
ax4.text(1.5, 162, 'Cp', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax4,data=Main_Dataset,x='cp',palette="dark:salmon_r")
ax4.set_xlabel("")
ax4.set_ylabel("")

# Fbs count
ax5.text(0.5, 290, 'Fbs', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5,data=Main_Dataset,x='fbs',palette="dark:salmon_r")
ax5.set_xlabel("")
ax5.set_ylabel("")

# Restecg count
ax6.text(0.75, 165, 'Restecg', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax6,data=Main_Dataset,x='restecg',palette="dark:salmon_r")
ax6.set_xlabel("")
ax6.set_ylabel("")

# Slp count
ax7.text(0.85, 155, 'Slope', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax7,data=Main_Dataset,x='slope',palette="dark:salmon_r")
ax7.set_xlabel("")
ax7.set_ylabel("")

# Thall count
ax8.text(1.2, 180, 'Thall', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax8.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax8,data=Main_Dataset,x='thal',palette="dark:salmon_r")
ax8.set_xlabel("")
ax8.set_ylabel("")

for s in ["top","right","left"]:
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)
    ax6.spines[s].set_visible(False)
    ax7.spines[s].set_visible(False)
    ax8.spines[s].set_visible(False)
# plt.show()

Features = Main_Dataset.drop(columns='target')
Features = pd.DataFrame(Features)

scaler = MinMaxScaler()
Norm_data = scaler.fit_transform(Features)
Norm_df = pd.DataFrame(Norm_data, columns= Features.columns)

desc_norm_df = describe(Norm_df)
print(desc_norm_df)

print(Norm_df.head(10))

X = Norm_df
y = Main_Dataset['target'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(X_train.shape)
print(y_train.shape)

training_acc_1 = []
test_acc_1 = []

range_k = range(3,20)

for number_k in range_k:
    knn = KNeighborsClassifier(n_neighbors = number_k, p=1)
    knn.fit (X_train, y_train.ravel())
    training_acc_1.append(knn.score(X_train,y_train))
    test_acc_1.append(knn.score(X_test, y_test))

# plt.figure(figsize=(15,5), dpi=100)    
# plt.plot(range_k, training_acc_1, label='Acc of training', color= 'black')
# plt.plot(range_k, test_acc_1, label='Acc of test set', color= '#E72B3B')
# plt.ylabel('Acc')
# plt.xlabel('Number of Neighbors')
# plt.title('Acc - Number of K')
# plt.legend()
# plt.xticks(range(1,20))
# plt.annotate('Best K_neighbor', xy=(3,0.89),xytext=(7.2,0.86), arrowprops=dict(facecolor='#E72B3B', shrink=0.05),fontsize=20)
# plt.axvline(x = 3, linestyle= 'dotted', c= 'black')
# plt.show()

best_k_index = test_acc_1.index(max(test_acc_1))
best_k = range_k[best_k_index]
print("Best K_neighbor:", best_k)

K = 3
clf_1 = KNeighborsClassifier(K, p=1)
clf_1.fit(X_train, y_train.ravel())
y_pred_1 = clf_1.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test,y_pred_1))

Best_knn = metrics.accuracy_score(y_test,y_pred_1)

conf_matrix_1 = confusion_matrix(y_test, y_pred_1)

colors = ["black", "#E72B3B", "#E72B3B", "#E72B3B"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
fig = plt.figure(figsize=(15, 3), dpi=200)
ax = plt.subplot()
plt.title("Confusion_matrix , KNN , K = 3 , P = 1")
annot = np.array([[f"{conf_matrix_1[0, 0]}", f"{conf_matrix_1[0, 1]}"],
                  [f"{conf_matrix_1[1, 0]}", f"{conf_matrix_1[1, 1]}"]], dtype=object)


sns.heatmap(conf_matrix_1,
            annot=annot,
            annot_kws={"size": 11},
            ax=ax,
            fmt='',
            cmap=cmap,
            cbar=True,
            )
# plt.xlabel("Pred")
# plt.ylabel("Real")
# plt.show()

XR = Main_Dataset.drop(columns='target')
yR = Main_Dataset['target']

rf = RandomForestClassifier()

rf.fit(XR, yR)

feature_importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': XR.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#E72B3B')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  
# plt.show()

# new_patient_data = {
#    'age': 60,
#     'sex': 1,  # Nam
#     'cp': 2,   # Đau thắt ngực không ổn định
#     'trestbps': 140,  # Huyết áp nghỉ
#     'chol': 260,      # Cholesterol huyết thanh (mg/dL)
#     'fbs': 1,         # Đường huyết nhị phân lớn hơn 120 mg/dL
#     'restecg': 0,     # Kết quả điện tâm đồ bình thường
#     'thalach': 120,   # Nhịp tim tối đa đạt được
#     'exang': 1,       # Đau thắt ngực gây ra bởi hoạt động
#     'oldpeak': 2.5,   # Giảm ST do vận động so với nghỉ
#     'slope': 1,       # phân đoạn ST liên quan đến sự gia tăng do tập thể dục
#     'ca': 1,          # động mạch vành
#     'thal': 3         # Thalassemia do cơ sở 3
# }

# scaled_data = scaler.transform([list(new_patient_data.values())])
# scaled_data_df = pd.DataFrame(scaled_data, columns=Features.columns)

# Dự đoán bệnh tim bằng mô hình KNN
# knn_prediction = clf_1.predict(scaled_data_df)

# print("KNN Prediction:", knn_prediction)

# if(knn_prediction[0] == 0):
#     print("The person does not have a heart disease")
# else:
#     print("The person has a heart disease")