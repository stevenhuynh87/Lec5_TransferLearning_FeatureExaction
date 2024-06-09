# import the necessary packages

from sklearn.metrics import classification_report
import numpy as np
import pickle
import os
from sklearn import svm
# Định nghĩa đường dẫn lưu model sau khi train, và chỉ ra nơi chứa file đặc trưng
# được trích xuất từ mạng Vgg16
BASE_CSV_PATH = "output"

def load_data_split(splitPath):
	# Khởi tạo danh sách chứa data và labels
	data = []
	labels = []
	# Duyệt qua các hàng trong file chứ đặc trưng và nhãn
	for row in open(splitPath):
		# Trích xuất nhãn và đặc trưng từ hàng của file chứa đặc trưng
		row = row.strip().split(",")
		label = row[0]
		features = np.array(row[1:], dtype="float")
		# Cập nhật danh sách data và label
		data.append(features)
		labels.append(label)
	# Chuyển data và labels sang arrays
	data = np.array(data)
	labels = np.array(labels)
	# Trả về tuple có hai thành phần data và labels
	return (data, labels)

# Khai báo đưowng dẫn đến các file *csv chứa đặc trưng
# để làm dữ liệu train và test cho bộ phân lớp SVM
trainingPath = os.path.sep.join([BASE_CSV_PATH,"training.csv"])
testingPath = os.path.sep.join([BASE_CSV_PATH,"evaluation.csv"])

# Chia tách phần data, nhãn để train và để test
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)

# Nạp file mã hóa label
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
le = pickle.loads(open(LE_PATH, "rb").read())

# Train bộ phân lớp SVM (model)
print("[INFO] training model...")
model = svm.SVC()
model.fit(trainX, trainY)

# đánh giá model
print("[INFO] evaluating...")
preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

# Lưu model thành file model.cpickle
print("[INFO] saving model...")
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])
f = open(MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()