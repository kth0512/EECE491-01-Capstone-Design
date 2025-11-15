import cv2
import numpy as np
from google.colab import files

def get_image_from_upload():
    print("이미지를 업로드해주세요.")
    uploaded = files.upload()
    if not uploaded:
        print("이미지가 업로드되지 않았습니다.")
        return None

    file_name = list(uploaded.keys())[0]
    img = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)
    return img