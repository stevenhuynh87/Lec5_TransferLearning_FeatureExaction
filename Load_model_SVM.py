# import the necessary packages
import numpy as np
import cv2
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Khởi tạo danh sách nhãn
classLabels = ["non_food", "food"]

# Nạp hình ảnh để thực hiện gắn nhãn (phân lớp)
print("[INFO] Đang nạp ảnh mẫu để phân lớp...")
imagePath ="image//0.jpg"
image = load_img(imagePath, target_size=(224, 224)) #thay đổi lại kích thước 224 x 224
image = img_to_array(image) # Chuyển ảnh lưu trữ bằng mảng
image = np.expand_dims(image, axis=0) # chuyển đổ dữ liệu theo quy định keras
image = preprocess_input(image) # Nhận dữ liệu
# Tạo danh sách để lưu dữ liệu ảnh
batchImages = []
batchImages.append(image)

# truyền ảnh qua mạng và đầu ra là các đặc trưng của ảnh,
# Sau đó định dạng lại các đă trưng đó thành một khối phẳng
model1 = VGG16(weights="imagenet", include_top=False)
batchImages = np.vstack(batchImages)
features = model1.predict(batchImages)
features = features.reshape((features.shape[0], 7 * 7 * 512))


# Nạp model (network) đã được train (pre-trained)
print("[INFO] Nạp model mạng pre-trained ...")
model = pickle.load(open('output//model.cpickle', 'rb'))

# Dự đoán nhãn (label) ảnh đầu vào. Ảnh được lưu trong data
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(features)

#Kết quả trả về: "0", "1": Tương ứng với food và non_food
if preds=="0":
	label="food"
else:
 	label="non_food"

# mở ảnh ảnh --> tạo label dự đoán trên ảnh --> Hiển thị ảnh
image = cv2.imread(imagePath)
# Tạo label dự đoán trên ảnh
cv2.putText(image, "label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)







