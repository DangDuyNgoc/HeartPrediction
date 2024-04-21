import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart.csv')

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
data.describe()

target_counts = data['target'].value_counts()
print(target_counts)

print("----------------------------------------------------------------")

X = data.drop(columns='target', axis=1)
Y = data['target']

NumDescripticeStats=data.describe(include = [np.number])
print(NumDescripticeStats)


feature_x='age'
feature_y='thal'
GroupByDF=data.loc[:,[feature_x, feature_y]].groupby(feature_x, as_index=False).mean()
# nhóm dữ liệu theo 'age' và tính giá trị trung bình của 'thal' cho mỗi nhóm.
print("----------------------------------------------------------------")
print(GroupByDF)

# plt.figure(figsize= (15,7))
# plt.scatter(GroupByDF[feature_x], GroupByDF[feature_y])
# plt.title(f'{feature_x} vs {feature_y} Plotting')
# plt.xlabel(feature_x, size=15)
# plt.ylabel(feature_y, size=15)
# plt.savefig(f'{feature_x} vs {feature_y} Plotting.jpg')
# plt.show()

# feature_x='age'
# feature_y='chol'
# GroupByDF=data.loc[:,[feature_x, feature_y]].groupby(feature_x, as_index=False).mean()
# print("----------------------------------------------------------------")
# print(GroupByDF)

m1 = GroupByDF[[feature_x]].values
m2 = GroupByDF[[feature_y]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Chọn và training data
model = RandomForestClassifier()
model.fit(X_train, y_train)

# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=4)
# X_poly = poly_reg.fit_transform(m1)
# print("----------------------------------------------------------------")
# print(X_poly[:5])

feature_x = 'age'
feature_y = 'thal'

GroupByDF = data.groupby(feature_x, as_index=False)[[feature_y]].mean()

X = GroupByDF[[feature_x]].values
y = GroupByDF[[feature_y]].values

model2 = RandomForestRegressor()
model2.fit(X, y)

# Predicting on the same X values used for training for demonstration purpose
y_prediction = model2.predict(X)

plt.figure(figsize=(15, 7))
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_prediction, color='red', label='Predicted Data')
plt.title(f'{feature_x} vs {feature_y} Plot')
plt.xlabel(feature_x, size=15)
plt.ylabel(feature_y, size=15)
plt.legend()
plt.show()


print("----------------------------------------------------------------")

# đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = pd.DataFrame({'age': [65],          # Tuổi của người
                        'sex': [1],           # Giới tính (1: Nam, 0: Nữ)
                        'cp': [3],            # Loại đau ngực (0-3)
                        'trestbps': [145],    # Huyết áp tĩnh (mm Hg)
                        'chol': [233],        # Cholesterol (mg/dl)
                        'fbs': [1],           # Đường huyết nhanh (1: > 120 mg/dl, 0: otherwise)
                        'restecg': [0],       # Kết quả điện tâm đồ nghỉ (0-2)
                        'thalach': [150],     # Nhịp tim tối đa đạt được
                        'exang': [0],         # Đau ngực gắng sức (1: Có, 0: Không)
                        'oldpeak': [2.3],     # Giảm ST gắng sức so với nghỉ
                        'slope': [0],         # Độ dốc ST gắng sức (0-2)
                        'ca': [0],            # Số mạch đã màu (0-3)
                        'thal': [1]}) 
prediction = model.predict(new_data)
if (prediction[0]== 0): 
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
# print("Predicted target:", prediction)