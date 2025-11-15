import cv2
import numpy as np
import torch

def draw_bounding_boxes(img, bounding_boxes):
    img_with_boxes = img.copy()
    if bounding_boxes:
        for box in bounding_boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_boxes
    
def render_tensor(ax, tensor, title=None):
    # Matplotlib 축(axis)에 텐서 이미지를 플로팅하는 함수수
    img_np = tensor.cpu().numpy()
    img_np = (img_np / 2.0) + 0.5
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(np.transpose(img_np, (1, 2, 0)))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)