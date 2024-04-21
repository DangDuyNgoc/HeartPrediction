import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('heart.csv')

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
data.describe()

target_counts = data['target'].value_counts()
print(target_counts)

# Tiền xử lý dữ liệu nếu cần

# Chia dữ liệu thành dữ liệu đầu vào (X) và nhãn (y)
X = data.drop(columns='target', axis=1)
y = data['target']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành dữ liệu huấn luyện và dữ liệu kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Các giá trị của n_neighbors để thử
n_neighbors_list = [3, 5, 7, 9]

# Duyệt qua từng giá trị của n_neighbors và tính độ chính xác
for n_neighbors in n_neighbors_list:
    # Huấn luyện mô hình KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Dự đoán trên dữ liệu kiểm tra
    y_pred = knn.predict(X_test)

    # Đánh giá độ chính xác của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Model with n_neighbors={n_neighbors}: {accuracy}")