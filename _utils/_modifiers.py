import numpy as np
import pandas as pd
import cv2
from PIL import Image


def get_corners(bbox):
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    corners = np.hstack((x1, y1, x1, y2, x2, y2, x2, y1))
    return corners


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    Parameters
    ----------
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    Returns
    -------
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    """
    x_ = corners[:, 0]
    y_ = corners[:, 1]
    xmin = min(x_)
    ymin = min(y_)
    xmax = max(x_)
    ymax = max(y_)
    enclosed_box = np.array([[xmin, ymin], [xmax, ymax]])
    return enclosed_box


def rotate_bboxes(rotated_img, bboxes_df, M):
    rotated_bboxes = []
    for bboxes in bboxes_df['bboxes']:
        corners = get_corners(bboxes).reshape(4, 2)
        newbb = []
        for i, coord in enumerate(corners):
            # Prepare the vector to be transformed
            v = [coord[0], coord[1], 1]
            # Perform the actual rotation and return the image
            calculated = np.dot(M, v)
            newbb.append([int(round(calculated[0])), int(round(calculated[1]))])
        enclosed_box = get_enclosing_box(np.array(newbb))
        rotated_bboxes.append(np.array(enclosed_box))
    bboxes_df['bboxes'] = pd.Series(rotated_bboxes, name='bboxes')
    return bboxes_df


def rotate_img(img, angle):
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    return (rotated_img, M)


def bboxes_to_anchor(bboxes_list):
    '''
    The shape received must be in (-1,4)
    '''
    cx = ((bboxes_list[:, 2])/2) + bboxes_list[:, 0]
    cy = ((bboxes_list[:, 3])/2) + bboxes_list[:, 1]
    w = bboxes_list[:, 2]
    h = bboxes_list[:, 3]
    anchor_boxes = np.array([cx, cy, w, h]).T
    return anchor_boxes


def anchor_to_bboxes(anchor_list):
    '''
    The shape received must be in (-1,4)
    '''
    x = anchor_list[:, 0] - anchor_list[:, 2]/2
    y = anchor_list[:, 1] - anchor_list[:, 3]/2
    w = anchor_list[:, 2]
    h = anchor_list[:, 3]
    bboxes_list = np.array([x, y, w, h]).T
    return bboxes_list


def convert_bboxes_to_xywh(bboxes_list):
    x = bboxes_list[:, 0]
    y = bboxes_list[:, 1]
    w = bboxes_list[:, 2] - bboxes_list[:, 0]
    h = bboxes_list[:, 3] - bboxes_list[:, 1]
    bboxes_converted = np.array([x, y, w, h]).T
    return bboxes_converted


def img_padding(img, pad_size=0, refill=False):
    '''
    Parameters
    img: np.array - an image array
    pad_size: int - value for padding
    refill: bool - If True, it will refill the border that contains less pixels until it becomes a squared image.
    Returns
    img: np.array
    '''
    # Check if the image is in grayscale or not
    if len(img.shape) == 2:
        if refill:
            img = np.pad(img, (pad_size, pad_size), mode='constant')
        else:
            img = np.pad(img, (pad_size, pad_size), mode='constant')
    else:
        # Consulta de tamanho de imagem
        if refill:
            channelpos = np.argmin(img.shape)
            max_idx = np.argmax(np.delete(img.shape, channelpos))
            min_idx = np.argmin(np.delete(img.shape, channelpos))
            diff_size = img.shape[max_idx] - img.shape[min_idx]
            img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
            if max_idx == 1:
                img = np.pad(img, ((0, diff_size), (0, 0), (0, 0)), mode='constant')
            else:
                img = np.pad(img, ((0, 0), (0, diff_size), (0, 0)), mode='constant')
        else:
            img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    return img


def rescale(img, percentage):
    if type(img) is np.ndarray:
        newsize = (np.array(img.shape[0:2]).astype(float) * percentage)
        return np.asarray(Image.fromarray(img).resize(np.flip(newsize).astype(int)))
    else:
        newsize = (np.array(np.asarray(img).shape[0:2]).T.astype(float) * percentage)
        return img.resize(np.flip(newsize).astype(int))

