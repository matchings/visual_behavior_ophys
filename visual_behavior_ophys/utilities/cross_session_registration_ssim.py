
# coding: utf-8

# @marinag

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#formatting
import seaborn as sns 
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
sns.set_context('notebook',font_scale=1.5,rc={'lines.markeredgewidth': 1.5})


from visual_behavior_ophys.utilities.lims_database import LimsDatabase


def get_lims_data(lims_id):
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    if lims_data.parent_session_id.values[0] is None: 
        lims_data['parent_session_id'] = lims_data.session_id.values[0]
    mouse_id = lims_data.external_specimen_id.values[0]
    lims_data.mouse_id = np.int(mouse_id)
    lims_data.ophys_session_dir = lims_data.datafolder.values[0][:-28]
    experiment_id = lims_data.lims_id.values[0]
    lims_data.insert(loc=2, column='experiment_id', value=experiment_id)
    del lims_data['datafolder']
    return lims_data


def get_average_image(lims_data):
    ophys_experiment_dir = os.path.join(lims_data.ophys_session_dir, 'ophys_experiment_' + str(lims_data.experiment_id.values[0]))
    processed_dir = os.path.join(ophys_experiment_dir, 'processed')
    segmentation_folder = [dir for dir in os.listdir(processed_dir) if 'ophys_cell_segmentation' in dir][0]
    segmentation_dir = os.path.join(processed_dir, segmentation_folder)
    image = mpimg.imread(os.path.join(segmentation_dir,'avgInt.tif'))
    return image


def get_lims_id_for_session_id(session_id, specimen_id):
    base_dir = r"\\allen\programs\braintv\production\neuralcoding\prod0\specimen_"
    data_dir = os.path.join(base_dir+str(specimen_id))
    session_dir = os.path.join(data_dir,'ophys_session_'+str(session_id))
    ophys_experiment_folder = [dir for dir in os.listdir(session_dir) if 'ophys_experiment' in dir][0]
    return ophys_experiment_folder[-9:]


def register_image(ref_image,image_to_shift):
    from skimage.feature import register_translation
    shift, error, diffphase = register_translation(ref_image, image_to_shift)
    from scipy.ndimage.interpolation import shift as scipy_shift
    shifted_image = scipy_shift(image_to_shift, shift, prefilter=False)
    return shifted_image


def get_overlay(parent_image,expt_image):
    rgb_image = np.empty((parent_image.shape[0],parent_image.shape[1],3))
    rgb_image[:,:,0] = parent_image
    rgb_image[:,:,1] = expt_image
    rgb_image[:,:,2] = 0
    return rgb_image


def get_avg_images(lims_data):
    expt_image = get_average_image(lims_data)
    parent_session_id = lims_data.parent_session_id.values[0]
    specimen_id = lims_data.specimen_id.values[0]
    parent_lims_id = get_lims_id_for_session_id(parent_session_id,specimen_id)
    parent_lims_data = get_lims_data(parent_lims_id)
    parent_image = get_average_image(parent_lims_data)
    shifted_expt_image = register_image(parent_image,expt_image)
    return parent_image, shifted_expt_image


def get_ssim(img0,img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1)
    return ssim_pair



if __name__ == '__main__':

    specimen_id = 599659785
    lims_id = 639932228

    # ## plot overlay & SSIM for single experiment, matched to parent session

    figsize=(20,5)
    fig,ax = plt.subplots(figsize=figsize)

    lims_data = get_lims_data(lims_id)
    parent_image, shifted_expt_image = get_avg_images(lims_data)
    img0 = parent_image
    img1 = shifted_expt_image

    ssim = get_ssim(img0, img1)
    overlay = get_overlay(img0, img1)
    label = str(lims_data.parent_session_id.values[0])+'-'+str(lims_id)+'\n SSIM: %.2f'
    ax.imshow(overlay)
    ax.set_title(label % (ssim))
    ax.axis('off')

    plt.show()