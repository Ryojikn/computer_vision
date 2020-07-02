import os
import json
import random
import secrets
import numpy as np
import pandas as pd
from PIL import Image
from itertools import product


def load_image(path, color='normal'):
    imgname = os.path.basename(path)
    if color == 'normal':
        img = Image.open(path)
    elif color == 'gray':
        img = Image.open(path).convert('L')
    return img, imgname


def load_image_as_data(path, color='all'):
    imgname = os.path.basename(path)
    if color == 'all':
        img = Image.open(path)
        imgarray = np.array(img)
    elif color == 'gray':
        img = Image.open(path).convert('P')
        imgarray = np.asarray(img)
    return imgarray, imgname


def read_regions_file(imagefilename, labels_path):
    label_json_file = imagefilename.split('.')[0]
    with open(labels_path + label_json_file + '.json') as jsonfile:
        labeljson = json.load(jsonfile)
    return labeljson


def get_bboxes(labeljson):
    '''
    Get bboxes from json file marked with VGG Image Annotator.
    Each jsonfile must contain annotation only for a single image.
    Parameters:
    Json containing annotations from VIA, in which the label attribute name must be marked as "Label"
    and the labels can receive any string.
    Returns:
    A pandas dataframe for this image with label | bboxes of that label in the following format: [top left x,y, bottom right x,y]
    P.S.: This function only works for rectangular bboxes at the moment.
    '''
    if len(labeljson.keys()) == 1:
        keyindex = list(labeljson.keys())[0]
        bboxes = []
        labels = []
        for label_mark in labeljson[keyindex]['regions']:
            actual_region = labeljson[keyindex]['regions']
            act_label_attr = actual_region[label_mark]['shape_attributes']
            curr_label = actual_region[label_mark]['region_attributes']['Label']
            if act_label_attr['name'] == 'rect':
                x1 = act_label_attr['x']
                y1 = act_label_attr['y']
                x2 = x1 + act_label_attr['width']
                y2 = y1 + act_label_attr['height']
                bboxes.append([[x1, y1], [x2, y2]])
                labels.append(curr_label)
    return pd.DataFrame({'label': labels, 'bboxes': bboxes})


def param_grid_search(config_dict, max_params):
    '''
    Parameters
    config_dict: dict - Dictionary with all Augmentation parameters. Must follow albumentation library list of parameters. For example:{'Blur': {'blur_limit': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}
    max_params: int - Max number of params to be applied in a sequence, must be below number of param variations that you're providing on config_dict.
    '''
    param_grid = {}
    for key in config_dict.keys():
        params_list = []
        items = sorted(config_dict[key].items())
        if not items:
            pass
        else:
            keys, values = zip(*items)
            for v in product(*values):
                output_dict = dict(zip(keys, v))
                params_list.append(output_dict)
        random.shuffle(params_list)
        param_grid[key] = params_list

    param_result_grid = []
    dict_size = len(param_grid[list(param_grid.keys())[0]])
    for index in range(0, dict_size):
        idxs = np.random.choice(range(max_params), max_params, replace=False)
        output_curr_grid = []
        for pos in idxs:
            dict_curr_key = list(param_grid.keys())[pos]
            rand_param = secrets.choice(param_grid[dict_curr_key])
            output_curr_grid.append({dict_curr_key: rand_param})
        param_result_grid.append(output_curr_grid)
    return param_result_grid

