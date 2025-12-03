import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def draw_bounding_boxes(img, bounding_boxes):
    img_with_boxes = img.copy()
    if bounding_boxes:
        for box in bounding_boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_boxes
    
def render_tensor(ax, tensor, title=None):
    # plot tensor image into Matplotlib (axis)
    img_np = tensor.cpu().numpy()
    img_np = (img_np / 2.0) + 0.5
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(np.transpose(img_np, (1, 2, 0)))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)
        
def plot_loss(num_epochs, train_loss_history, val_loss_history):  
    # plot train loss, val loss vs epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss',
            linestyle='--', color='green', marker='o', markersize=4)
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss',
            linestyle='-', color='blue', marker='x', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (1 - SSIM)')
    plt.title(r'Training and Validation Loss (SSIM Loss)')
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.show()