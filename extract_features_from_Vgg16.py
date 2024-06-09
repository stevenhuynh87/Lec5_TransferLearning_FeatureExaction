# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import pickle
import random
import os

# tải mạng VGG16 đã được huấn luyện trước với Dataset imagenet
# loại bỏ các lớp FC và bộ phân lớp ảnh (tham số include_top=False)
# và khởi tạo bộ mã hóa nhãn
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None

# Định nghĩa các folder
TRAIN = "training"   # Chứa dữ liệu để train
TEST = "evaluation"  # Chứa dữ liệu để evaluation, đánh giá model sau khi train
VAL = "validation" # chứa dữ liệu validation, đánh giá model trong quá trình train (sau 1 epoch)
BASE_PATH = "Dataset" # Chứa các folder training,evaluation,validation
BASE_CSV_PATH = "output" # chứa các file trích xuất đặc trưng, file nhãn (*.csv), file model sau khi train

BATCH_SIZE = 32  # Kích thước mini-batch

# Lặp qua dữ liệu để chia tách dữ liệu train, validation, evaluation để trích xuất đặc trưng và tạo file chứa nhãn lớp tương ứng
for split in (TRAIN, TEST, VAL):
	# lấy tất cả các đường dẫn chứa hình ảnh để tách
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([BASE_PATH, split])
	imagePaths = list(paths.list_images(p))
	# xáo trộn ngẫu nhiên các đường dẫn hình ảnh
	# và sau đó trích xuất nhãn lớp từ folder chứa file ảnh
	random.shuffle(imagePaths)
	labels = [p.split(os.path.sep)[-2] for p in imagePaths]
	# nếu bộ mã hóa nhãn là Không có, hãy tạo nó (tức là, tạo bộ mã hóa nhãn)
	if le is None:
		le = LabelEncoder()
		le.fit(labels)
	# mở file CSV để ghi dữ liệu (nhãn lớp)
	csvPath = os.path.sep.join([BASE_CSV_PATH,"{}.csv".format(split)])
	csv = open(csvPath, "w")

	# lặp lại các hình ảnh trong mỗi mini-batch
	for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
		# trích xuất từng mini-batch hình ảnh và nhãn, sau đó khởi tạo danh sách
		# hình ảnh thực tế để được truyền qua mạng Vgg16 để trích xuất đặc trưng
		print("[INFO] processing batch {}/{}".format(b + 1,int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
		batchPaths = imagePaths[i:i + BATCH_SIZE]
		batchLabels = le.transform(labels[i:i + BATCH_SIZE])
		batchImages = []
		# Duyệt qua các ảnh và nhãn trong mini-bacth hiện hành
		for imagePath in batchPaths:
			# Nạp hình ảnh đầu vào đồng thời thay đổi kích thước thành 224x224 pixel
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)
			# Tiền xử lý hình ảnh bằng cách: (1) mở rộng kích thước và
			# (2) trừ cường độ điểm ảnh RGB trung bình khỏi tập dữ liệu ImageNet
			image = np.expand_dims(image, axis=0)
			image = preprocess_input(image)
			# Sau khi xử lý xong thêm ảnh vào mini-batch
			batchImages.append(image)

		# truyền hình ảnh qua mạng và kết quả đầu ra làm các đặc trưng
		# sau đó định hình lại các đặc trưng thành một khối phẳng
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=BATCH_SIZE)
		features = features.reshape((features.shape[0], 7 * 7 * 512))

		# Duyêt qua nhãn lớp và các đặc trưng được trích xuất
		for (label, vec) in zip(batchLabels, features):
			# Lưu trữ nhãn và đặc trưng vào file
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(label, vec))
	# Đóng CSV file
	csv.close()
# Lưu nhãn lớp vào vào file le.cpickle
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
