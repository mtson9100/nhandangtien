from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys

cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name = ['khong co tien ','10000','20000','200000','500000']
#get 1 model vgg16
def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input') # dau vao la 1 anh 128*128*3
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax', name='predictions')(x) # dung theo so luong dau ra

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

## Load weights model da train
my_model = get_model()
my_model.load_weights("weights-39-0.96.hdf5")

while(True):
    # Capture frame-by-frame
    ##doc lien tuc anh tu camera

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    ## Resize anh ve co 182*128
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    ## Convert thành tensor
    image = np.expand_dims(image, axis=0)

    ## Predict (đưa vào trong mạng để dự đoán)
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0])) # hien thi ten cac lop
    print(np.max(predict[0],axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):##  nếu như đự đáon tren 80% một mạnh giá nào đó và có mang cầm tiền trên tay thì sẽ hiển thị mệnh giá tiền lên


        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2
# phong chu hien thi
        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org) # chay camera voi ten la picture

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # -cv2.waitKey () trả về giá trị số nguyên 32 Bit (có thể phụ thuộc vào nền tảng). Đầu vào khóa là ASCII, là một giá trị số nguyên 8 Bit.
        # Vì vậy, bạn chỉ quan tâm đến 8 bit này và muốn tất cả các bit khác bằng 0. Điều này bạn có thể đạt được với:cv2.waitKey(1) & 0xFF
        break

# When everything done, release the capture
cap.release()  # phát hành tài nguyên phần mềm
cv2.destroyAllWindows() # giải phóng tài nguyên phần cứng

