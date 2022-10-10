import numpy as np
import cv2
import time
import os

# Label: 00000 là ko cầm tiền, còn lại là các mệnh giá
label = "50000"

cap = cv2.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i=0
while(True):
    # Capture frame-by-frame
    # đọc liên tục từ camera
    i+=1
    ret, frame = cap.read()  #-Cap.read () trả về một giá trị bool (Đúng / Sai).Nếu khung được đọc đúng, nó sẽ là True. Vì vậy, bạn có thể

    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None,fx=0.3,fy=0.3)#cv2.Resize, hay gọi cách khác là scale ảnh, là việc ta chỉnh kích thước ảnh về kích thước mới (có thể giữ tỉ lệ ảnh ban đầu hoặc không).

    # Hiển thị
    cv2.imshow('frame',frame)

    # Lưu dữ liệu
    if i>=60 and i<=1060:# lấy ảnh từ bức 60 tới 1060
        print("Số ảnh capture = ",i-60)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('data/' + str(label)):#Sử dụng "path.exists" bạn có thể nhanh chóng kiểm tra xem một tệp hoặc thư mục có tồn tại hay không
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png",frame) #v2.imwrite()được sử dụng để lưu hình ảnh vào bất kỳ thiết bị lưu trữ nào. Thao tác này sẽ lưu hình ảnh theo định dạng được chỉ định trong thư mục làm việc hiện tại.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #-cv2.waitKey () trả về giá trị số nguyên 32 Bit (có thể phụ thuộc vào nền tảng). Đầu vào khóa là ASCII, là một giá trị số nguyên 8 Bit.
        #Vì vậy, bạn chỉ quan tâm đến 8 bit này và muốn tất cả các bit khác bằng 0. Điều này bạn có thể đạt được với:cv2.waitKey(1) & 0xFF
        break

# When everything done, release the capture
cap.release() # phát hành tài nguyên phần mềm
cv2.destroyAllWindows() # giải phóng tài nguyên phần cứng