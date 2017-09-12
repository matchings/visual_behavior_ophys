
# @marinag

# code for filtering roi masks using metrics provided in objectlist.txt 
# metrics are described here: http://confluence.corp.alleninstitute.org/display/IT/Ophys+Segmentation  

# In[1]:

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage.measurements as measurements
#from aibs.Analysis.InDevelopment.plotTools import saveFigure 

#formatting
import seaborn as sns 
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.5) 
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False


def saveFigure(fig, fname, formats = ['.pdf'],transparent=False,**kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])
    else:
        fig.set_size_inches(11,8.5)
    for f in formats:
        fig.savefig(fname + f, transparent = transparent, orientation = 'landscape')


# plot original, unfiltered multi page masks (maxInt_masks2.h5)
def plot_masks(masks):
    fig,ax=plt.subplots(2,2)
    ax=ax.ravel()
    n = 0
    for i in range(masks.shape[0]):
        page = masks[i]
        labels,n_rois = measurements.label(page)
        n = n+n_rois
        ax[i].imshow(labels,cmap='jet')
        ax[i].grid(False)
        ax[i].set_title(str(n_rois)+' rois')
        ax[i].axis('off')
    plt.tight_layout()
    print str(n)+' rois'
    return masks
   
# In[13]:

# make dictionary of individual roi masks for all cells across pages of the original mask
# goes through every index in objectlist and uses centroid position and page within original mask to find the corresponding individual roi mask for that index
def make_roi_dict(df,masks,index=' traceindex'):
    roi_dict={}
    for roi in df[index].values:
        roi_mask = np.zeros(masks[0,:,:].shape)
        cx = df[df[index]==roi][' cx'].values[0]
        cy = df[df[index]==roi][' cy'].values[0]
        p = df[df[index]==roi][' mask2Frame'].values[0]
        page = masks[p]
        labels, n_rois = measurements.label(page)
        roi_id = labels[cy,cx]
        inds = np.where(labels==roi_id)
        roi_mask[inds]=1
        roi_dict[roi] = roi_mask
    return roi_dict

def make_roi_mask_array(df,roi_dict):
    roi_mask_array = np.empty(len(roi_dict.keys()),roi_dict[roi_dict.keys()[0]].shape)
    for i,roi in enumerate(roi_mask_dict.keys()):
        roi_mask_array[i] = roi_mask_dict[roi]
    return roi_dict

def get_roi_metrics(expt_dir):
    roi_metrics_df = pd.read_csv(os.path.join(expt_dir, 'objectlist.txt'))  # segmentation metrics
    roi_metrics_df = roi_metrics_df[roi_metrics_df[' traceindex'] != 999]
    return roi_metrics_df


def get_original_roi_masks(expt_dir):
    g = h5py.File(os.path.join(expt_dir, "maxInt_masks2.h5"))  # mask excluding edge cells
    original_roi_masks = np.asarray(g['data'])  # original, unfiltered masks
    return original_roi_masks


def make_roi_dict_from_metrics_df(roi_metrics_df, original_roi_masks):
    roi_mask_dict = {}
    i = 0
    for roi in roi_metrics_df[' traceindex'].values:
        roi_mask = np.zeros(masks[0, :, :].shape)
        cx = roi_metrics_df[roi_metrics_df[' traceindex'] == roi][' cx'].values[0]
        cy = roi_metrics_df[roi_metrics_df[' traceindex'] == roi][' cy'].values[0]
        p = roi_metrics_df[roi_metrics_df[' traceindex'] == roi][' mask2Frame'].values[0]
        page = original_roi_masks[p]
        labels, n_rois = measurements.label(page)
        roi_id = labels[cy, cx]
        inds = np.where(labels == roi_id)
        roi_mask[inds] = 1
        if np.sum(
                roi_mask) > 1000:  # happens when CoM of roi isnt actually in the roi, for example when an roi contains 2 cells, the CoM lies between and falls on the background
            pass
        else:
            roi_mask_dict[i] = roi_mask
            i += 1
    return roi_mask_dict


def get_roi_metrics_for_roi_mask_dict(roi_metrics_df, roi_mask_dict):
    trace_index_list = []
    for roi in roi_mask_dict.keys():
        try:
            labels, n_rois = measurements.label(roi_mask_dict[roi])
            c = measurements.center_of_mass(labels)
            cy = c[0]
            cx = c[1]
            # to deal with values ending in .5 being rounded down instead of up
            if str(cx).endswith('.5'):
                cx = cx + 0.1
            if str(cy).endswith('.5'):
                cy = cy + 0.1
            cy = np.int(np.round(cy))
            cx = np.int(np.round(cx))
            trace_index = roi_metrics_df[(roi_metrics_df[' cy'] == cy) & (roi_metrics_df[' cx'] == cx)][' traceindex'].values[0]
            trace_index_list.append(trace_index)
        except:  # if trace_ind cant be found in object_df, skip that roi
            print 'could not find roi',str(roi),'from dict in in roi_metrics_df'
        roi_metrics = roi_metrics_df[roi_metrics_df[' traceindex'].isin(trace_index_list)]
    return roi_metrics



#plot individual roi mask
def plot_roi_mask(roi_mask_dict,roi,max_image=None,ax=None):
    if ax is None: 
        fig,ax=plt.subplots()
    if max_image is not None:
        plt.imshow(max_image,cmap='gray',vmin=0,vmax=1)
    mask = np.empty(roi_dict[roi].shape)
    mask[np.where(roi_dict[roi]==0)]=np.nan
    mask[np.where(roi_dict[roi]==1)]=1
    ax.imshow(mask,cmap='jet',alpha=0.4,vmin=0,vmax=1)
    ax.axis('off')
    ax.grid(False)
    if ax is not None: 
        return ax


# plot distribution of metric values across all rois
def plot_metric_distribution(df,metric,thresh=None,ax=None):
    if ax is None: 
        fig,ax=plt.subplots()
    ax.hist(df[metric].values,bins=50);
    ax.set_title('metric = '+metric+', threshold = '+str(thresh))
    ax.set_xlabel('value')
    ax.set_ylabel('# rois')
    if thresh:
        ax.axvline(thresh,color='g',linestyle='dashed',label='thresh_min')
#     ax.legend(loc=9)
    if ax is not None: 
        return ax


# get roi indices with metric values within a given range, for a single metric
def get_filtered_inds_single_metric(roi_metrics,metric,thresh=None):
    filtered_inds = roi_metrics[(roi_metrics[metric]>thresh)][' traceindex'].values
    return filtered_inds


# create single plane mask for rois in thresh_inds as defined above
def make_filtered_mask(roi_mask_dict,filtered_inds):
    filtered_mask = np.empty(roi_mask_dict[1].shape)
    filtered_mask[:]= np.nan
    for roi in roi_mask_dict.keys():
        if roi in filtered_inds:
            inds = np.where(roi_mask_dict[roi]==1)
            if len(inds[0])>10000: # ignore cases where entire background is picked up due to weird shaped roi having centroid on background
                pass
            else:
                filtered_mask[inds] = 1
        else: 
            inds = np.where(roi_mask_dict[roi]==1)
            if len(inds[0])>10000: # ignore cases where entire background is picked up due to weird shaped roi having centroid on background
                pass
            else:
                filtered_mask[inds] = 0
    return filtered_mask


# plot thresholded single plane mask on max projection image, optional input for labels & save params
def plot_filtered_mask(roi_mask_dict,filtered_mask,max_image=None,labels=False,ax=None,save_dir=False,title=None,show=False):
    roi_mask_dict_copy = roi_mask_dict.copy()
    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    if max_image is not None: 
        ax.imshow(max_image,cmap='gray',vmin=0,vmax=np.amax(max_image)/2)
        ax.imshow(filtered_mask,cmap='jet',alpha=0.3,vmin=0,vmax=1)
        color = 'white'
    else: 
        cax = ax.imshow(filtered_mask,cmap='jet',vmin=0,vmax=1)
        color = 'black'
    if labels is not False:
        for i,roi in enumerate(labels): 
            tmp = roi_mask_dict_copy[i]
            [x1,y1]=np.where(tmp==1)
            x = np.mean(x1)
            y = np.mean(y1)
            ax.text(y,x,str(roi),color=color,fontsize=16)
    # plt.colorbar(cax,shrink=0.8);
#    ax.set_title('red = included, blue = excluded')
    ax.grid(False)
    ax.axis('off')
    if save_dir is True:
        saveFigure(fig,os.path.join(save_dir,title),formats = ['.png'],size=(10,10))
    plt.close()
    if ax is not None: 
        return ax


# generate figure with 1) distribution of metric values & 2) mask of filtered rois on max projection image
def plot_metric_thresh(roi_metrics,roi_mask_dict,metric,thresh,max_image,save_dir=False):
    fig_title = str(expt_id)+'_'+str(metric)+'_'+str(thresh)
    figsize=(20,7)
    fig,ax = plt.subplots(1,2,figsize=figsize)
    ax = ax.ravel()
    plot_metric_distribution(roi_metrics,metric,thresh=thresh,ax=ax[0])
    filtered_inds = get_filtered_inds_single_metric(roi_metrics,metric,thresh=thresh);
    filtered_mask = make_filtered_mask(roi_mask_dict,filtered_inds)
    plot_filtered_mask(roi_mask_dict,filtered_mask,max_image,ax=ax[1])
    if save_dir:
        saveFigure(fig,os.path.join(save_dir,fig_title),formats = ['.png'],size=figsize)
    plt.close(fig)


def plot_filtered_roi_masks_for_range_of_filter_metrics(roi_metrics, roi_mask_dict, filter_params, max_image, save_dir=None):
    save_dir = os.path.join(save_dir, 'segmentation_metrics')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    metrics = [' area', ' shape0', ' meanInt0']
    am = filter_params[' area']
    mm = filter_params[' shape0']
    sm = filter_params[' meanInt0']
    filters = {' area': np.arange(am - 20, am + 70, 20), ' shape0': np.arange(sm - 0.1, sm + 0.4, 0.1),
               ' meanInt0': np.arange(mm - 20, mm + 60, 20)}
    for metric in metrics:
        for filter in filters[metric]:
            plot_metric_thresh(roi_metrics, roi_mask_dict, metric, filter, max_image, save_dir)

 
# get roi indices that meet multiple metric parameters   
def get_filtered_inds_for_standard_metrics(roi_metrics,filter_params):
    metrics = filter_params.keys()
    inds_list = []
    for i,metric in enumerate(metrics):
        inds = get_filtered_inds_single_metric(roi_metrics,metric,thresh=filter_params[metric])
        inds_list = np.hstack((inds_list,inds))
    return inds_list
       

def get_standard_filter_parameters(zoom,layer=None):
    if zoom == 3:
        filter_params = {' area': 200, ' meanInt0': 40, ' shape0': 0.3}
    elif zoom == 2 and layer == 5:
        filter_params = {' area': 160, ' meanInt0': 40, ' shape0': 0.3}
    elif zoom == 2:
        filter_params = {' area': 120, ' meanInt0': 25, ' shape0': 0.2}
    elif zoom == 1:
        filter_params = {'area': 10, ' meanInt0': 30, ' shape0': 0.1}
    return filter_params


# get roi indices for rois that meet all metric filter criteria
def get_filtered_indices(roi_metrics,filter_params):
    inds_list = get_filtered_inds_for_standard_metrics(roi_metrics,filter_params)
    filtered_indices = []
    for roi in roi_metrics[' traceindex'].values:
        if roi in inds_list: #and (roi in maxMeanRatio)
            filtered_indices.append(roi)
    filtered_indices = np.asarray(filtered_indices)
    return filtered_indices
  



# In[80]:
       
if __name__ == '__main__':
    
    zoom = 2
    layer = 5
    expt_dir = r'\\aibsdata2\nc-ophys\ImageData\Technology\Marina\161020-M253178\DoC'
    expt_id = expt_dir.split('\\')[-2]
    save_dir = os.path.join(expt_dir,'analysis')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    f = h5py.File(os.path.join(expt_dir,"maxInt_masks2.h5")) #mask excluding edge cells, corresponding to objectlist classification
    original_roi_mask = f['data'] #original, unfiltered multi page masks
    max_image = mpimg.imread(os.path.join(expt_dir,'maxInt_a13.png'))
    #objectlist - metrics associated with segmentation masks, used for filtering
    roi_metrics = pd.read_csv(os.path.join(expt_dir,'objectlist.txt')) #segmentation metrics
    roi_metrics = roi_metrics[roi_metrics[' traceindex']!=999] # remove edge cells, etc. maxInt_masks2 has rois with traceindex = 999 removed
      
 
    #standard filter params - for zoom settings on Sutter MoM - selected using output of plot_seg_filters below
    filter_params = get_standard_filter_parameters(zoom,layer)
    
    # make dictionary of masks where keys are roi IDs and values are the binary masks for that roi
    roi_mask_dict = make_roi_dict(roi_metrics,original_roi_mask)
    # makes & saves figures for a range of different filter params - useful in deciding what filter_params settings to use for your FOV, depth etc
    plot_filtered_roi_masks_for_range_of_filter_metrics(roi_metrics,filter_params,roi_mask_dict,max_image,save_dir)

    ### generate new set of filtered masks (roi_dict)
    #get indices of objects in objectlist that meet filter criteria
    filtered_inds = get_filtered_indices(roi_metrics,filter_params['area_min'],filter_params['shape_min'],filter_params['meanInt_min'])
    #create single plane mask containing only rois meeting filter criteria
    filtered_mask = make_filtered_mask(unfiltered_roi_dict,filtered_inds)
    #make new roi_dict for filtered indices
    roi_dict = {}
    for i,roi in enumerate(filtered_inds): 
        roi_dict[i] = unfiltered_roi_dict[roi]
    #plot filtered masks labeled with original objectlist IDs
    plot_filtered_mask(roi_mask_dict,filtered_mask,max_image,labels=filtered_inds,save_dir=save_dir,title='filtered_masks_labeled_objectlist')
    #plot filtered mask with new IDs 
    plot_filtered_mask(roi_mask_dict,filtered_mask,max_image,labels=roi_dict.keys(),save_dir=save_dir,title='filtered_masks_labeled')
    #plot filtered mask with no labels
    plot_filtered_mask(roi_mask_dict,filtered_mask,max_image,labels=False,save_dir=save_dir,title='filtered_masks')


        
        
        