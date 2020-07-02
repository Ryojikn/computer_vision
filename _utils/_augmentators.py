from albumentations.augmentations.transforms import *
from PIL import Image
import secrets
import uuid
import copy
import _loaders as load
import _modifiers as mod


def image_augmentation(img, img_name, params_list, bbox_rotation=False, labels_path=None, save=False, output_path=None):
    '''
    Parameters:
    img: np.array - img array
    img_name: str - img filename
    config_dict: dict - Dictionary containing all parameters that you'd like to be used to for augmentation possibilities.
    max_params: int - Max number of params to be applied in a sequence.
    n_imgs: int - Number of generated images desired per image, must be bellow config_dict max possibilities.
    bbox_rotation: bool - True if you'd like to also rotate bboxes
    labels_path: str - Provide the path for region files marked with VGG Image Annotator v1.0.6
    save: bool - True if you'd like to save your augmented images in any place
    output_path: str - Path for saving augmented images
    '''
    bbox_enabled, rotated_bboxes = False, None
    rotateflag = False
    rotation_angle = 0

    img_name_2 = img_name.split('.')[0]
    mask_img = Image.fromarray(img)
    temp_img = copy.deepcopy(img)
    temp_params = copy.deepcopy(secrets.choice(params_list))
    for params in temp_params:
        key = list(params.keys())[0]
        params = list(params.values())[0]
        if ('rotate' in key.lower()):
            rotateflag = True
            params['img'] = temp_img
            rotation_angle = params['angle']
            temp_img, M = mod.rotate_img(**params)
            temp_mask = Image.new('L', mask_img.size, 255)
            temp_mask = temp_mask.rotate(-params['angle'], expand=True)
            temp_mask = temp_mask.resize((temp_img.shape[1], temp_img.shape[0]))
            if bbox_rotation:
                labeljson = load.read_regions_file(img_name, labels_path)
                bboxes_df = load.get_bboxes(labeljson)
                rotated_bboxes = mod.rotate_bboxes(img, bboxes_df, M)
                bbox_enabled = True
        else:
            try:
                params['img'] = temp_img
                temp_img = eval(key)().apply(**params)
            except:
                del params['img']
                params['image'] = temp_img
                temp_img = eval(key)().apply(**params)
    if not rotateflag:
        temp_mask = Image.new('L', mask_img.size, 255)
    newname = img_name_2 + '_' + uuid.uuid4().hex + '.jpg'
    if save:
        save_path = output_path + newname + '.jpg'
        transformed_image = Image.fromarray(temp_img)
        transformed_image.save(save_path)
    if bbox_rotation and bbox_enabled:
        bboxes_df = rotated_bboxes
        return temp_img, newname, temp_mask, bboxes_df, rotation_angle
    else:
        return temp_img, newname, temp_mask, rotation_angle

