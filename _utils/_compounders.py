import copy
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image


def check_image_size(img):
    if type(img) == list:
        temp_img = img
    else:
        temp_img = np.asarray(img)
    height = temp_img.shape[0]
    width = temp_img.shape[1]
    return width, height


def image_pasting(bgimg, img, xy_origin=None, mask=None):
    '''
    Parameters:
    bgimg: np_array
    img: np_array
    xy_origin: tuple - x_origin and y_origin coordinates for the image to be pasted
    '''
    temp_img = copy.deepcopy(Image.fromarray(bgimg))
    temp_img.paste(Image.fromarray(img), xy_origin, mask)
    temp_img = np.asarray(temp_img)
    return temp_img


def image_random_pasting(bgimg, img, mask=None):
    '''
    Parameters:
    bgimg: np_array
    img: np_array
    xy_origin: tuple - x_origin and y_origin coordinates for the image to be pasted
    '''
    bgw, bgh = check_image_size(bgimg)
    imgw, imgh = check_image_size(img)
    maxw_orig = bgw - imgw
    maxh_orig = bgh - imgh
    x = np.random.randint(0, maxw_orig, 1)
    y = np.random.randint(0, maxh_orig, 1)
    xy_origin = (x, y)
    temp_img = copy.deepcopy(Image.fromarray(bgimg))
    temp_img.paste(Image.fromarray(img), xy_origin, mask)
    temp_img = np.asarray(temp_img)
    return temp_img, xy_origin


def plot_bboxes(img, bboxes, bboxes_label=None, shape='rect'):
    if shape == 'rect':
        drawing = copy.deepcopy(img)
        if type(bboxes) == pd.core.frame.DataFrame:
            for idx in bboxes.index:
                curr_label = bboxes['label'][idx]
                curr_bbox = bboxes['bboxes'][idx]
                top_left = tuple(curr_bbox[0])
                bottom_right = tuple(curr_bbox[1])
                drawing = cv2.putText(drawing, curr_label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), lineType=cv2.LINE_AA)
                drawing = cv2.rectangle(drawing, top_left, bottom_right, (0, 255, 0), 2)
            return drawing
        else:
            bboxes = bboxes.reshape(-1, 4)
            for i, bbox in enumerate(bboxes):
                if (bboxes_label is not None):
                    curr_label = bboxes_label[i]
                    drawing = cv2.putText(drawing, curr_label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), lineType=cv2.LINE_AA)
                curr_bbox = bbox.reshape(-1, 2)
                top_left = tuple(curr_bbox[0])
                bottom_right = tuple(curr_bbox[1])
                if type(top_left[0]) is torch.Tensor:
                    top_left = tuple(np.array(top_left).astype(int))
                    bottom_right = tuple(np.array(bottom_right).astype(int))
                if type(drawing) is torch.Tensor:
                    drawing = np.array(drawing)
                drawing = cv2.rectangle(drawing, top_left, bottom_right, (0, 255, 0), 2)
            if type(drawing) is cv2.UMat:
                drawing = drawing.get()
            return drawing
    elif shape == 'poly':
        drawing = copy.deepcopy(img)
        for idx in bboxes.index:
            curr_label = bboxes['label'][idx]
            curr_bbox = bboxes['bboxes'][idx]
            top_left = tuple(curr_bbox[0])
            print(top_left)
            drawing = cv2.putText(drawing, curr_label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), lineType=cv2.LINE_AA)
            drawing = cv2.polylines(drawing, [curr_bbox], True, (0, 255, 0), thickness=0.4)
        return drawing

