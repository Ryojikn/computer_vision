from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import pandas as pd
import json
import joblib
import copy
import numpy as np
import os
import _loaders as load
import _modifiers as modif
import _compounders as compound
import _augmentators as augment
import secrets


class DataGen():
    '''Data Generator
    This generator for now only works if you're providing a dataset that contains imgs or backgrounds with 3 channels, at least one of them must be colored.
    Parameters:
    source_dict -  {'images': img_path_list, 'backgrounds': bgs_path_list}
    config_dict -  config_dict based on Albumentation functions.
    max_params: int - how many parameters should be used in a row for each generated image.
    dataset_size: int - how many images should be generated
    labels_path: str - where are located the files with the labels
    bbox_rotation: bool - If rotate is provided in config_dict, it should be True
    resize: tuple - define the shape that you'd like to have as output imgs.
    output_path: str - location where the whole dataset must be wrote to.
    '''
    def __init__(self, source_dict, config_dict, max_params, dataset_size,  labels_path, rescale_list = None, bbox_rotation=False, resize=None, output_path=''):
        ## Starting data
        self.imgs_path_list = source_dict['images']
        self.bgs_path_list = source_dict['backgrounds']
        self.config_dict = config_dict
        self.max_params = max_params
        self.bbox_rotation = bbox_rotation
        self.labels_path = labels_path
        self.resize = resize
        self.rescale = rescale_list

        # Loaded data for work
        self.img = None
        self.img_name = None
        self.bg = None
        self.bg_name = None
        self.bboxes = None
        self.param_grid = None


        self.rotation_angle = 0
        self.bboxes_flat = None
        self.xy_origin = None
        self.target = None
        self.target_label = None

        ## Output data
        self.final_features = []
        self.final_target = []
        self.final_names = []


        ## Functions triggered at beggining
        self.totaldocs = self.__get_docs__()
        self.ohenc_model = self.__ohenc__()
        self.__generate_img__(dataset_size)
        self.__save_imgs__(dataset_size, output_path)

    def __get_docs__(self):
        doc_list = [''.join(x for x in os.path.splitext(os.path.basename(i))[0] if x.isalpha()) for i in self.imgs_path_list]
        return np.unique(doc_list)

    def __ohenc__(self):
        labels = []
        for doc in self.totaldocs:
            temp_img_name = np.random.choice([x for x in self.imgs_path_list if doc in x])
            temp_img_name = os.path.basename(temp_img_name)
            tempjson = load.read_regions_file(temp_img_name, self.labels_path)
            temp_bboxes = load.get_bboxes(tempjson)
            labels.append(temp_bboxes['label'])
        labels = pd.concat(labels).tolist()
        labels = list(map(lambda el: [el], labels))
        ohenc = OneHotEncoder().fit(labels)
        joblib.dump(ohenc, 'one_hot_model.jbl')
        return ohenc

    def __generate_img__(self, dataset_size):
        '''
        This function will generate a batch of the size of dataset size, doesn't actually matter how long is
        this dataset size, as long as you have space in disk to place the images at the end.
        '''
        for i in range(0, dataset_size):
            # Loading information
            self.__load_imgs__()
            self.__load_bboxes__()
            self.__load_params__()

            ## Creating augmentation dataset
            self.__img_augmentation__()
            self.__bbox_resizing__()

            # Composing data for saving the generated images.
            self.__compose_target__()
            features = np.array(self.img.T)
            self.final_features.append(features)
            self.final_target.append(self.target)

    def __load_imgs__(self):
        '''
        Loading images and backgrounds
        '''
        self.img, self.img_name = load.load_image_as_data(np.random.choice(self.imgs_path_list))
        self.bg, self.bg_name = load.load_image_as_data(np.random.choice(self.bgs_path_list))

    def __load_bboxes__(self):
        '''
        Loading bounding boxes using VGG Image Annotator 1.0.6 as marking tool.
        The bboxes must be labeled as "Label" and the output should be in the format .json provided by the tooling.
        '''
        tempjson = load.read_regions_file(self.img_name, self.labels_path)
        self.bboxes = load.get_bboxes(tempjson)

    def __load_params__(self):
        '''
        Performs and provides a param_grid pipeline that will be applied to your images, considering the max_params parameter.
        '''
        self.param_grid = load.param_grid_search(self.config_dict, self.max_params)

    def __img_augmentation__(self):
        '''
        Performs the augmentation pipeline
        '''
        if self.bbox_rotation:
            newimg, newimgname, mask, self.bboxes, self.rotation_angle = augment.image_augmentation(self.img, self.img_name, self.param_grid, bbox_rotation=self.bbox_rotation, labels_path=self.labels_path)
        else:
            newimg, newimgname, mask, self.rotation_angle = augment.image_augmentation(self.img, self.img_name, self.param_grid, bbox_rotation=self.bbox_rotation, labels_path=self.labels_path)
        self.final_names.append(newimgname)

        if self.rescale is not None and type(self.rescale) is list:
            newscale = secrets.choice(self.rescale)
            newimg = modif.rescale(newimg, newscale)
            mask = modif.rescale(mask, newscale)
            newseries = self.bboxes['bboxes'].apply(lambda x: np.array(x)) * newscale
            newseries = newseries.apply(lambda x: np.array(x).astype(int))
            self.bboxes['bboxes'] = newseries

        ## Proportional images shape implementation
        imgs_proportion = min(np.array(newimg.shape)/np.array(self.bg.shape))
        if (imgs_proportion < 1):
            self.bg = np.array(Image.fromarray(self.bg).resize(size=np.flip(np.array(newimg.shape[0:2]) + 100)))
        self.__bg_augmentation__()
        self.img, self.xy_origin = compound.image_random_pasting(self.bg, newimg, mask)
        for col in self.bboxes.loc[:, self.bboxes.columns != 'label'].columns:
            for i, line in enumerate(self.bboxes[col]):
                newx = np.array(line)[:, 0]
                another_x = newx + self.xy_origin[0]
                newy = np.array(line)[:, 1]
                another_y = newy + self.xy_origin[1]
                self.bboxes[col][i] = np.vstack([another_x, another_y]).T

    def __compose_target__(self):
        test_df = pd.DataFrame({'label': self.bboxes['label']})
        bbox_labels = self.ohenc_model.transform(test_df)
        self.target_label = self.ohenc_model.inverse_transform(bbox_labels)

        ## Top left, bottom right bboxes
        bboxes = self.bboxes_flat.reshape(-1, 4)

        ## Converting to xywh and not top left, bottom right
        # bboxes = convert_bboxes_to_xywh(bboxes)

        ## Converting to anchor boxes
        # bboxes = bboxes_to_anchor(bboxes)

        bbox_labels_out = []
        for bbox in bbox_labels:
            bbox_labels_out.append(bbox.toarray()[0])

        #one hot encoded
        # rows = len(bbox_labels_out)
        # cols = 4+len(self.ohenc_model.categories_[0])
        # bboxes_target = np.zeros((rows, cols))
        # bboxes_target[:, 0:4] = bboxes
        # bboxes_target[:, 4: cols] = bbox_labels_out

        #simple labels
        rows = len(bbox_labels_out)
        cols = 4 ## bboxes
        bboxes_target = np.zeros((rows, cols))
        bboxes_target[:, 0:4] = bboxes

        self.target = [self.rotation_angle, bboxes_target, self.target_label]

    def normalizedata(self, bboxes, maxsize, minsize):
        normalized = []
        for val in bboxes:
            curr_val = (val-minsize)/(maxsize-0)
            if curr_val > 1:
                curr_val = 1
            elif curr_val < 0:
                curr_val = 0
            normalized.append(curr_val)
        return normalized

    def __bg_augmentation__(self):
        customdict = copy.deepcopy(self.config_dict)
        try:
            if 'Rotate' or 'rotate' in customdict:
                try:
                    customdict.pop('Rotate')
                except:
                    customdict.pop('rotate')
        except:
            pass
        finally:
            bg_paramgrid = load.param_grid_search(customdict, self.max_params - 1)
            newbg, bgname, bgmask, bg_rot_angle = augment.image_augmentation(self.bg, self.bg_name, bg_paramgrid)
            self.bg = newbg

    def __bbox_resizing__(self):
        flattened_bboxes = np.concatenate(self.bboxes['bboxes'].values.flatten()).flatten()
        self.img = modif.img_padding(self.img, pad_size=2, refill=True)
        if self.resize is not None:
            proportion = np.array(self.resize) / np.array(self.img.shape[0:1])
            proportion = max(proportion)
            self.img = np.array(Image.fromarray(self.img).resize(self.resize))
            flattened_bboxes = np.concatenate((self.bboxes['bboxes'] * proportion).values.flatten()).flatten().astype(int)
        self.bboxes_flat = flattened_bboxes

    def __save_imgs__(self, dataset_size, output_path):
        for i in range(0, dataset_size):
            output_img = Image.fromarray(self.final_features[i].T)
            output_img.save(output_path + self.final_names[i])

            docname = self.final_names[i][:-4]
            angle = str(self.final_target[i][0])
            docface = str(self.final_target[i][1].tolist()[-2:])
            docface_labels = json.dumps(self.final_target[i][2].reshape(1, -1)[0][-2:].tolist())
            bboxes = str(self.final_target[i][1].tolist()[:-2])
            bboxes_labels = json.dumps(self.final_target[i][2].reshape(1, -1)[0][:-2].tolist())

            curr_json = "{\"name\": \"" + docname + "\", " + "\"rot_angle\": " + angle + ", " + "\"docface\": " + docface + ", " + "\"docface_labels\": " + docface_labels + ", " + "\"bboxes\": " + bboxes + ", " + "\"bboxes_labels\": " + bboxes_labels + "}"

            output_json = json.loads(curr_json)
            target_path = output_path + "labels/" + self.final_names[i][:-4] + ".json"
            with open(target_path, 'w') as f:
                json.dump(output_json, f)


class Dataset(data.Dataset):
    'Caracteriza um dataset para o Pytorch'
    def __init__(self, imgs_path_list, labels_path):
        self.imgs_path_list = imgs_path_list
        self.labels_path = labels_path
        self.dataset_size = len(imgs_path_list)
        self.img = None
        self.img_name = None
        self.rot_angle = 0

    def __len__(self):
        return self.dataset_size

    def __load_img__(self, idx):
        self.img, self.img_name = load.load_image_as_data(self.imgs_path_list[idx])

    def __load_labels__(self, idx):
        label_filename = os.path.basename(self.imgs_path_list[idx])[:-4] + ".json"
        with open(self.labels_path + label_filename, 'r') as f:
            label_json = json.load(f)
        self.rot_angle = label_json['rot_angle']
        self.docface = label_json['docface']
        self.bboxes = label_json['bboxes']

    def __getitem__(self, idx):
        self.__load_img__(idx)
        self.__load_labels__(idx)

        return self.img.T, self.rot_angle, np.array(self.docface), np.array(self.bboxes)
