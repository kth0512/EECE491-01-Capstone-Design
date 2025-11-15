import cv2

def draw_bounding_boxes(img, bounding_boxes):
    img_with_boxes = img.copy()
    if bounding_boxes:
        for box in bounding_boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_boxes