import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

def get_image_from_upload():
    print("테스트할 이미지를 업로드해주세요.")
    uploaded = files.upload()
    if not uploaded:
        print("이미지가 업로드되지 않았습니다.")
        return None

    file_name = list(uploaded.keys())[0]
    img = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)
    return img

def draw_bounding_boxes(img, bounding_boxes):
    img_with_boxes = img.copy()
    if bounding_boxes:
        for box in bounding_boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_boxes