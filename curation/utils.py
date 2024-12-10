import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pydicom
import dicom2nifti
import ants # env to use: reg-36

os.environ["PATH"] += ':/netopt/rhel7/versions/ANTs/2.2/antsbin/bin:'

dicom2nifti.settings.disable_validate_slice_increment() 
dicom2nifti.settings.enable_resampling()
dicom2nifti.settings.set_resample_spline_interpolation_order(3)  # spline interpolation order (0 nn , 1 bilinear, 3 cubic)
dicom2nifti.settings.set_resample_padding(-1000) #-1000 for CT data background


"""
-------------------------------------------------------------------------------------------------------
-------------------------------Utility functions of RCC CT data curation-------------------------------
-------------------------------------------------------------------------------------------------------
"""


### -------------------------------MISC FUNCTIONS-------------------------------
def import_dicom_header(dicom_dir):
    # pulls out first dicom and reads in header using pydicom
    dicom1 = os.listdir(dicom_dir)
    dicom1_dir = dicom_dir + dicom1[0]
    metadata = pydicom.dcmread(dicom1_dir, stop_before_pixels=True)
    return metadata


def replace_item(list, prev, new):
    return [new if a==prev else a for a in list]


def remove_item(list, item):
    return [a for a in list if a!=item]


def convert_dicom_to_nifti(dicom_dir, save_path, flip=0):
    dicom2nifti.dicom_series_to_nifti(dicom_dir, save_path, reorient_nifti=True) 
    nii_image = nib.load(save_path)
    rot_image = np.rot90(nii_image.get_fdata()) #rotate by 90 once before saving, for prone also rotates to not-prone
    nii_image.set_data_dtype(np.float64)
    new_img = nib.Nifti1Image(rot_image, affine=nii_image.affine, header=nii_image.header) #save rotated image with previous header info
    if flip:
        new_img = new_img.slicer[:, :, ::-1] #flip in z axis to match dicom data
    nib.save(new_img, save_path)


def phase_from_label(label):
    art_substrings = ["art", "ART", "arterial"]
    pv_substrings = ["pv", "PV"]
    noncon_substrings = ["nc", "noncon", "NC"]
    del_substrings = ["del", "delay", "DEL"]

    if any(substring in label for substring in art_substrings):
        phase = "arterial"
    elif any(substring in label for substring in pv_substrings):
        phase = "portven"
    elif any(substring in label for substring in noncon_substrings):
        phase = "noncon"
    elif any(substring in label for substring in del_substrings):
        phase = "delay"
    else:
        phase = []

    return phase 

def get_phase(dir, **kwargs):
    inc_noncon = kwargs.get('inc_noncon', None)

    filelist = os.listdir(dir)
    phslist = [l.split('.')[0] for l in filelist]
    phslist = [l.split('_')[0] for l in phslist]
    phslist = [l.split('-')[0] for l in phslist]
    if inc_noncon:
        #to_rem = {"BB", "tumor", "kidney", "left", "right"}
        to_inc = {"noncon", "delay", "portven", "arterial"}
    else:
        #to_rem = {"BB", "tumor", "noncon", "kidney", "left", "right", "dont"}
        to_inc = {"delay", "portven", "arterial"}
    #phslist = [l for l in phslist if l not in to_rem]
    phslist = [l for l in phslist if l in to_inc]
    phslist = [*set(phslist)]
    return phslist


def get_best_mask(dir, masktag, reg_stat):

    #find phases that are registered (only considers registered masks!!)
    del reg_stat["pid"]
    del reg_stat["mask"]
    phs = [k for k, v in reg_stat.items() if v>1]
    
    BB_masks = []
    BB_masks_markers = []
    for ph in phs:
        BB_masks += glob.glob(os.path.join(dir,("BB*"+ph+"*_reg.nii.gz")))
        BB_masks_markers += glob.glob(os.path.join(dir,("BB*"+ph+"*_markers*_reg.nii.gz")))

    if masktag=="cropped":
        BB_masks = BB_masks + glob.glob(os.path.join(dir,"BB*noncon*_cropped.nii.gz"))
        BB_masks_markers = BB_masks_markers + glob.glob(os.path.join(dir,"BB*noncon*_markers*_cropped.nii.gz"))
    elif masktag=="not_cropped":
        BB_masks = BB_masks + glob.glob(os.path.join(dir,"BB_*_noncon.nii.gz"))
        BB_masks_markers = BB_masks_markers + glob.glob(os.path.join(dir,"BB_*_noncon_markers_1.nii.gz"))

    tumor_masks =[]
    for ph in phs:
        tumor_masks += glob.glob(os.path.join(dir,("tumor*"+ph+"*_reg.nii.gz")))
    
    BB_only_masks = [m for m in BB_masks if m not in BB_masks_markers]

    if tumor_masks:
        mask = tumor_masks[0]
    else:
        if BB_only_masks:
            mask = BB_only_masks[0]
        else:
            try:
                mask = BB_masks_markers[0] #should I pick based on phase? 
            except:
                print("no mask found")
                pass
    #print(mask)
    return mask

def get_mask_fname(dir, phase, **kwargs):

    suffix = kwargs.get('suffix', "")

    phase_masks = [f for f in os.listdir(dir) if phase in f]
    phase_masks = [m for m in phase_masks if "BB" in m or "tumor" in m]
    #print(phase_masks)
    
    if phase_masks:
        if "tumor_R_" + phase + suffix + ".nii.gz" in phase_masks:
            mask = "tumor_R_" + phase + suffix + ".nii.gz"
            test = mask.replace("_R_","_L_")
        elif "tumor_L_" + phase + suffix + ".nii.gz" in phase_masks:
            mask = "tumor_L_" + phase + suffix + ".nii.gz"
            test = mask.replace("_L_","_R_")
        elif "BB_R_" + phase + suffix + ".nii.gz" in phase_masks:
            mask = "BB_R_" + phase + suffix + ".nii.gz"
            test = mask.replace("_R_","_L_")
        elif "BB_L_" + phase + suffix + ".nii.gz" in phase_masks:
            mask = "BB_L_" + phase + suffix + ".nii.gz"
            test = mask.replace("_L_","_R_")
        elif "BB_R_" + phase + "_markers_1" + suffix + ".nii.gz" in phase_masks:
            mask = "BB_R_" + phase + "_markers_1" + suffix + ".nii.gz"
            test = mask.replace("_R_","_L_")
        elif "BB_L_" + phase + "_markers_1" + suffix + ".nii.gz" in phase_masks:
            mask = "BB_L_" + phase + "_markers_1" + suffix + ".nii.gz"
            test = mask.replace("_L_","_R_")
        else:
            mask = phase_masks[0]
    else:
        mask = "none.nii.gz"
        test = []
        print('no mask, phase:', phase)
    
    if test in phase_masks:
        mask = []
        print('both left and right tumors - combine')
    
    #if suffix:
    #    mask = mask[:-7] + suffix + '.nii.gz'
        
    return mask


def get_nicknames(phase):
    if phase == 'noncon':
       list = 'noncon|concon|NC'
    elif phase == 'delay':
        list= 'del|delay|DEL'
    elif phase == 'arterial':
        list = 'art|arterial|ART'
    elif phase == 'portven':
        list = 'pv|portven|PV'
    else:
        print("unknown phase:", phase)
        list = ''
    return list


### -------------------------------ANNOTATION FUNCTIONS-------------------------------


def create_mask(mask_function, phase_img, phase_path, curr_anntn_list, phase_img_header, output_filename, prone, phase):
    #general function that sets up converstion to mask and saves mask and visualizes it

    mask = np.zeros_like(phase_img.get_fdata())
    ns = mask.shape[2]

    mask, slice_num = mask_function(curr_anntn_list, mask, ns)
    
    if prone:
        mask = np.flip(mask, 0)
        mask = np.flip(mask, 1)
        #mask = np.rot90(mask, k=1, axes=(0,1))
    new_img = nib.Nifti1Image(mask, affine=phase_img.affine, header=phase_img_header) #save mask with phase header info
    nib.save(new_img, output_filename)  
    
    #print([ns-slice_num+3, ns-slice_num+2, ns-slice_num+1, ns-slice_num])
    print([slice_num, slice_num+1, slice_num+2, slice_num+3])
    #visualize_reg_mask(nii_path, output_filename, [ns-slice_num+3, ns-slice_num+2, ns-slice_num+1, ns-slice_num])
    visualize_reg_mask(phase_path, output_filename, [slice_num, slice_num+1, slice_num+2, slice_num+3])
    #visualize_reg_mask(nii_path, output_filename, [ns-slice_num+1, ns-slice_num, ns-slice_num-1, ns-slice_num-2])
    visualize_reg_mask(phase_path, output_filename, [int(ns/2)-2, int(ns/2)-1, int(ns/2), int(ns/2)+1])


def vertices_to_mask_slice(anntns_list, mask, ns):
    # for polygon annotations converts vertices to mask
    
    for idx,val in anntns_list.iterrows():
        vertices = eval(val['data.vertices'])
        #slice_num = val['Instance Number']
        slice_num = ns - val['Instance Number']

        nx, ny = mask[:,:,slice_num].shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
    
        path = Path(vertices)
        grid = path.contains_points(points)
        mask2D = grid.reshape((ny,nx))
        mask[:,:,slice_num] = mask2D.astype(int)
    
    return mask, slice_num


def BB_multislice_to_mask(anntns_list, mask, ns):
    # for multislice bounding box annotations
    
    for idx,val in anntns_list.iterrows():
        x, y = val['data.y'], val['data.x']
        ht, wth = val['data.width'], val['data.height']
        #slice_num = val['Instance Number']
        slice_num = ns - val['Instance Number']
        mask[round(x):round(x+wth),round(y):round(y+ht),slice_num] = int(1) #indices must be integers
    
    return mask, slice_num


def BB_and_markers_to_mask_1(anntns_list, mask, ns):
    # one way of dealing with annotations that have one central BB and a top and bottom marker
    # this assumes the BB is the same size across slices
    # linearly interpolates the center of the BB across slices

    BB_anntn = anntns_list[anntns_list['labelName'].str.contains('BB_')]
    centop_anntn = anntns_list[anntns_list['labelName'].str.contains('cen_top_')]
    cenbot_anntn = anntns_list[anntns_list['labelName'].str.contains('cen_bot_')]
    #print(centop_anntn[['data.x', 'data.y', 'Instance Number']])

    #x_c1, y_c1, z0 = BB_anntn['data.y'].values[0], BB_anntn['data.x'].values[0], BB_anntn['Instance Number'].values[0]
    x_c1, y_c1, z0 = BB_anntn['data.y'].values[0], BB_anntn['data.x'].values[0], ns-BB_anntn['Instance Number'].values[0]
    ht, wth = BB_anntn['data.width'].values[0], BB_anntn['data.height'].values[0]
    x0, y0 = x_c1 + (wth/2), y_c1 + (ht/2)
    #xt, yt, zt = centop_anntn['data.y'].values[0], centop_anntn['data.x'].values[0], centop_anntn['Instance Number'].values[0]
    #xb, yb, zb = cenbot_anntn['data.y'].values[0], cenbot_anntn['data.x'].values[0], cenbot_anntn['Instance Number'].values[0]
    xt, yt, zt = centop_anntn['data.y'].values[0], centop_anntn['data.x'].values[0], ns-centop_anntn['Instance Number'].values[0]
    xb, yb, zb = cenbot_anntn['data.y'].values[0], cenbot_anntn['data.x'].values[0], ns-cenbot_anntn['Instance Number'].values[0]
    # fill in center, top, bottom BB\n",
    print(x0, y0, z0)
    print(xt, yt, zt)
    print(xb, yb, zb)
    mask[int(round(x0-wth/2)):int(round(x0+wth/2)),int(round(y0-ht/2)):int(round(y0+ht/2)),z0] = int(1) 
    mask[int(round(xt-wth/2)):int(round(xt+wth/2)),int(round(yt-ht/2)):int(round(yt+ht/2)),zt] = int(1) 
    mask[int(round(xb-wth/2)):int(round(xb+wth/2)),int(round(yb-ht/2)):int(round(yb+ht/2)),zb] = int(1) 

    #interpolate center and fill in rest of remaining slices between top and bottom
    if zb > zt:
        for z in range(zt+1,z0):
            t = (z-z0)/ (zt-z0)
            x, y = (xt-x0)*t + x0, (yt-y0)*t + y0
            mask[int(round(x-wth/2)):int(round(x+wth/2)),int(round(y-ht/2)):int(round(y+ht/2)),z] = int(1) 

        for z in range(z0+1,zb):
            t = (z-z0)/ (zb-z0)
            x, y = (xb-x0)*t + x0, (yb-y0)*t + y0
            mask[int(round(x-wth/2)):int(round(x+wth/2)),int(round(y-ht/2)):int(round(y+ht/2)),z] = int(1)
    else:
        for z in range(zb+1,z0):
            t = (z-z0)/ (zb-z0)
            x, y = (xt-x0)*t + x0, (yt-y0)*t + y0
            mask[int(round(x-wth/2)):int(round(x+wth/2)),int(round(y-ht/2)):int(round(y+ht/2)),z] = int(1) 

        for z in range(z0+1,zt):
            t = (z-z0)/ (zt-z0)
            x, y = (xb-x0)*t + x0, (yb-y0)*t + y0
            mask[int(round(x-wth/2)):int(round(x+wth/2)),int(round(y-ht/2)):int(round(y+ht/2)),z] = int(1)

    return mask, z0


def BB_and_markers_to_mask_2(anntns_list, mask, ns):
    # 2nd way of dealing with annotations that have one central BB and a top and bottom marker
    # computes corners of the BB in top, center and bottom tumor slice (assuming BB is the same size as the center BB)
    # uses min and max values to compute largest 3D bounding box (will overestimate most likely but will probably encompass full tumor)

    BB_anntn = anntns_list[anntns_list['labelName'].str.contains('BB_')]
    centop_anntn = anntns_list[anntns_list['labelName'].str.contains('cen_top_')]
    cenbot_anntn = anntns_list[anntns_list['labelName'].str.contains('cen_bot_')]

    #x_c1, y_c1, z0 = BB_anntn['data.y'].values[0], BB_anntn['data.x'].values[0], BB_anntn['Instance Number'].values[0]
    x_c1, y_c1, z0 = BB_anntn['data.y'].values[0], BB_anntn['data.x'].values[0], ns-BB_anntn['Instance Number'].values[0]
    ht, wth = BB_anntn['data.width'].values[0], BB_anntn['data.height'].values[0]
    #xt, yt, zt = centop_anntn['data.y'].values[0], centop_anntn['data.x'].values[0], centop_anntn['Instance Number'].values[0]
    #xb, yb, zb = cenbot_anntn['data.y'].values[0], cenbot_anntn['data.x'].values[0], cenbot_anntn['Instance Number'].values[0]
    xt, yt, zt = centop_anntn['data.y'].values[0], centop_anntn['data.x'].values[0], ns-centop_anntn['Instance Number'].values[0]
    xb, yb, zb = cenbot_anntn['data.y'].values[0], cenbot_anntn['data.x'].values[0], ns-cenbot_anntn['Instance Number'].values[0]

    c0 = [(x_c1, y_c1, z0), (x_c1, y_c1+ht, z0), (x_c1+wth, y_c1, z0), (x_c1+wth, y_c1+ht, z0)]
    ct = [(xt-wth/2, yt-ht/2, zt), (xt+wth/2, yt-ht/2, zt), (xt-wth/2, yt+wth/2, zt), (xt+wth/2, yt+ht/2, zt)]
    cb = [(xb-wth/2, yb-ht/2, zb), (xb+wth/2, yb-ht/2, zb), (xb-wth/2, yb+wth/2, zb), (xb+wth/2, yb+ht/2, zb)]

    xmax, ymax, zmax = np.stack((np.array(c0), np.array(ct), np.array(cb)), axis=0).max(axis=(0,1))
    xmin, ymin, zmin = np.stack((np.array(c0), np.array(ct), np.array(cb)), axis=0).min(axis=(0,1))

    if zb > zt:
        mask[int(round(xmin)):int(round(xmax)),int(round(ymin)):int(round(ymax)),zt:zb+1] = int(1) 
    else: 
        mask[int(round(xmin)):int(round(xmax)),int(round(ymin)):int(round(ymax)),zb:zt+1] = int(1) 
    return mask, z0


### -------------------------------REGISTRATION FUNCTIONS-------------------------------

def crop_nifti_z(nifti_path, index_1, index_2):
    nii_image = nib.load(nifti_path)
    new_img = nii_image.slicer[:,:,index_1:index_2]
    nib.save(new_img, nifti_path)


def find_midz(mask_img):
    #find center, min + max in z of tumor using mask
    
    slices_mask = np.unique(np.nonzero(mask_img)[2])
    midz = ((slices_mask.max() - slices_mask.min()) // 2) + slices_mask.min()
    minz = slices_mask.min()
    maxz = slices_mask.max()
    return int(minz), int(midz), int(maxz)


def auto_register_z(image_list, mask_list, d):
    
    ns = [nib.load(im).shape[2] for im in image_list]
    range = np.array(ns) * np.array(d)

    #center tumor
    center = np.zeros_like(range)
    mins = np.zeros_like(range)
    maxs = np.zeros_like(range)
    for m, mask in enumerate(mask_list):
        mask_np = nib.load(mask).get_fdata()
        slices_mask = np.unique(np.nonzero(mask_np)[2])
        center[m] = ((slices_mask.max() - slices_mask.min()) // 2) + slices_mask.min()
        mins[m] = slices_mask.min()
        maxs[m] = slices_mask.max()
    
    # determine phase with smallest range -- crop rest to match that phase
    ind = np.argmin(range)
    midz = center[0]
    below = np.abs(center - mins)
    above = np.abs(maxs - center)

    for i, image in enumerate(image_list):
        mask_nii = mask_list[i]
        phs_image = nib.load(image)
        phs_mask_img =nib.load(mask_nii)

        if i != ind: #dont crop image with smallest range
            top_slices = int(((ns[ind] -center[ind]) * d[ind]) // d[i])
            bot_slices = int((center[ind] * d[ind]) // d[i])
            print(top_slices, center[i], bot_slices)

            cent = int(center[i])
            phs_image = phs_image.slicer[:,:,max(0,cent-bot_slices):cent+top_slices]    
            phs_mask_img = phs_mask_img.slicer[:,:,max(0,cent-bot_slices):cent+top_slices]   
            if i==0:
                midz = bot_slices

        phase_nii_new = image[:-7] + '_cropped.nii.gz'
        phs_mask_nii_new = mask_nii[:-7] + '_cropped.nii.gz'
        nib.save(phs_image, phase_nii_new)
        nib.save(phs_mask_img, phs_mask_nii_new)
    
    print("minz, midz, maxz:", midz-above[0], midz, midz+below[0])

    return int(midz-above[0]), int(midz), int(midz+below[0])


def fix_reg_nslice(nifti_to_fix, noncon_nii, dir, n):
    nii_image = nib.load(nifti_to_fix)
    nii_z_max = nii_image.shape[2]
    
    if dir == 'up':
        crop_nifti_z(nifti_to_fix, 0, nii_z_max-n)
        crop_nifti_z(noncon_nii, n, nii_z_max) #crop noncon to match size of phase
    elif dir == 'down':
        crop_nifti_z(nifti_to_fix, n, nii_z_max)
        crop_nifti_z(noncon_nii, 0, nii_z_max-n) #crop noncon to match size of phase


def register_nifti_affine(noncon_nii, phase_nii, phs_mask_nii):
    noncon = ants.image_read(noncon_nii)
    phase = ants.image_read(phase_nii)
    phsmask = ants.image_read(phs_mask_nii)

    #txfile = ants.affine_initializer(noncon, phase)
    #phase = ants.apply_transforms(fixed=noncon, moving=phase, transformlist=[txfile])
    #phsmask = ants.apply_transforms(fixed=noncon, moving=phsmask, transformlist=[txfile])

    mytx = ants.registration(fixed=noncon, moving=phase, type_of_transform= 'Affine', verbose=True)
    phs_mask_reg = ants.apply_transforms(fixed=noncon, moving=phsmask, transformlist=mytx['fwdtransforms']) # register mask as well using same transforms

    noncon2 = noncon-noncon.min()
    overlay = mytx['warpedmovout']-mytx['warpedmovout'].min()
    ants.plot(image=noncon2, overlay=overlay, cmap = 'Greys_r', overlay_cmap = 'jet', overlay_alpha = 0.3 )
    #ants.plot(image=noncon2, overlay=overlay, cmap = 'Greys_r', overlay_cmap = 'jet', overlay_alpha = 0.3, axis =2 )

    return mytx['warpedmovout'], phs_mask_reg


def register_nifti_syn(noncon_nii, phase_nii, phs_mask_nii):
    noncon = ants.image_read(noncon_nii)
    phase = ants.image_read(phase_nii)
    phsmask = ants.image_read(phs_mask_nii)
    # resample?
    #fi = ants.resample_image(, (60,60), 1, 1)
    #phase = ants.resample_image(phase, (512,512,93), 1, 1)
    #txfile = ants.affine_initializer(noncon, phase)
    #tx = ants.read_transform(txfile, dimension=2)
    #noncon2 = ants.apply_transforms(fixed=noncon, moving=phase, transformlist=[txfile])


    mytx = ants.registration(fixed=noncon, moving=phase, type_of_transform= 'SyN', verbose=True)
    phs_mask_reg = ants.apply_transforms(fixed=noncon, moving=phsmask, transformlist=mytx['fwdtransforms']) # register mask as well using same transforms

    noncon2 = noncon-noncon.min()
    overlay = mytx['warpedmovout']-mytx['warpedmovout'].min()
    ants.plot(image=noncon2, overlay=overlay, cmap = 'Greys_r', overlay_cmap = 'jet', overlay_alpha = 0.3 )
    ants.plot(image=noncon2, overlay=overlay, cmap = 'Greys_r', overlay_cmap = 'jet', overlay_alpha = 0.3, axis =2 )

    return mytx['warpedmovout'], phs_mask_reg


### -------------------------------VISUALIZATION FUNCTIONS-------------------------------

def view_reg_path(path_1, path_2, loc, **kwargs):
        savepath = kwargs.get('savepath', None)

        nii_img = nib.load(path_1)
        header = nii_img.header
        print(header.get_data_shape())
        image1 = nii_img.get_fdata()

        nii_img = nib.load(path_2)
        header = nii_img.header
        print(header.get_data_shape())
        image2 = nii_img.get_fdata()

        view_registration(image1, image2, loc, savepath=savepath)


def view_registration(image1, image3, loc, **kwargs):
    
    savepath = kwargs.get('savepath', None)

    mid1 = int(image1.shape[2]/2) #plot middle slice

    if not loc:
        loc = [mid1, mid1]

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(figsize=(20,10), ncols=5, nrows=3)  

    img1 = ax1.imshow(image1[:, :, loc[0]-2], cmap="Greys_r", vmin=-200, vmax=800)
    ax1.set_title('Axial Slice 1')
    #fig.colorbar(img1, ax=ax1)

    img2 = ax2.imshow(image1[:, :, loc[0]-1], cmap="Greys_r", vmin=-200, vmax=800)
    ax2.set_title('Axial Slice 2')
    #fig.colorbar(img2, ax=ax2)

    img3 = ax3.imshow(image1[:, :, loc[0]], cmap="Greys_r", vmin=-200, vmax=800)
    ax3.set_title('Axial Slice 3')
    #fig.colorbar(img3, ax=ax3)

    img4 = ax4.imshow(image1[:, :, loc[0]+1], cmap="Greys_r", vmin=-200, vmax=800)
    ax4.set_title('Axial Slice 4')
    #fig.colorbar(img4, ax=ax4)

    img5 = ax5.imshow(image1[:, :, loc[0]+2], cmap="Greys_r", vmin=-200, vmax=800)
    ax5.set_title('Axial Slice 5')
    #fig.colorbar(img4, ax=ax4)

    img6 = ax6.imshow(image3[:, :, loc[1]-2], cmap="Greys_r", vmin=-200, vmax=800)
    ax6.set_title('Axial Slice 1')
    #fig.colorbar(img9, ax=ax9)

    img7 = ax7.imshow(image3[:, :, loc[1]-1], cmap="Greys_r", vmin=-200, vmax=800)
    ax7.set_title('Axial Slice 2')
    #fig.colorbar(img10, ax=ax10)

    img8 = ax8.imshow(image3[:, :, loc[1]], cmap="Greys_r", vmin=-200, vmax=800)
    ax8.set_title('Axial Slice 3')
    #fig.colorbar(img11, ax=ax11)

    img9 = ax9.imshow(image3[:, :, loc[1]+1], cmap="Greys_r", vmin=-200, vmax=800)
    ax9.set_title('Axial Slice 4')
    #fig.colorbar(img12, ax=ax12)

    img10 = ax10.imshow(image3[:, :, loc[1]+2], cmap="Greys_r", vmin=-200, vmax=800)
    ax10.set_title('Axial Slice 4')
    #fig.colorbar(img12, ax=ax12)

    ax11.imshow(image1[:, :, loc[0]-2], cmap="Greys_r", vmin=-200, vmax=800)
    ax11.imshow(image3[:, :, loc[1]-2], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    ax11.set_title('Overlay Slice 1')

    ax12.imshow(image1[:, :, loc[0]-1], cmap="Greys_r", vmin=-200, vmax=800)
    ax12.imshow(image3[:, :, loc[1]-1], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    ax12.set_title('Overlay Slice 2')

    ax13.imshow(image1[:, :, loc[0]], cmap="Greys_r", vmin=-200, vmax=800)
    ax13.imshow(image3[:, :, loc[1]], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    ax13.set_title('Overlay Slice 3')

    ax14.imshow(image1[:, :, loc[0]+1], cmap="Greys_r", vmin=-200, vmax=800)
    ax14.imshow(image3[:, :, loc[1]+1], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    ax14.set_title('Overlay Slice 4')

    ax15.imshow(image1[:, :, loc[0]+2], cmap="Greys_r", vmin=-200, vmax=800)
    ax15.imshow(image3[:, :, loc[1]+2], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    ax15.set_title('Overlay Slice 4')

    plt.show()

    if savepath:
        fig.savefig(savepath)


def view_reg_mask_path(nc_path, phase_path_reg, nc_mask_path, phs_mask_path_reg, loc, **kwargs):

    savepath = kwargs.get('savepath', None)

    nii_img = nib.load(nc_path)
    header = nii_img.header
    print(header.get_data_shape())
    nc_image = nii_img.get_fdata()
    nii_img = nib.load(nc_mask_path)
    nc_mask = nii_img.get_fdata()

    
    nii_img = nib.load(phase_path_reg)
    header = nii_img.header
    print(header.get_data_shape())
    phs_image_reg = nii_img.get_fdata()
    nii_img = nib.load(phs_mask_path_reg)
    phs_mask_reg = nii_img.get_fdata()

    view_registration_with_mask(nc_image, phs_image_reg, nc_mask, phs_mask_reg, loc, savepath=savepath)


def view_registration_with_mask(nc_image, phs_image_reg, nc_mask, phs_mask_reg, loc, **kwargs):
#visualize 5 axial slices with mask overlaid

    savepath = kwargs.get('savepath', None)

    mid1 = int(nc_image.shape[2]/2) #plot middle slice
    if not loc:
        loc = [mid1, mid1]

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(figsize=(20,10), ncols=5, nrows=3)   

    ax1.imshow(nc_image[:, :, loc[0]-2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax1.imshow(nc_mask[:, :, loc[0]-2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax1.set_title('Noncontrast')
    #fig.colorbar(img1, ax=ax1)

    ax2.imshow(nc_image[:, :, loc[0]-1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax2.imshow(nc_mask[:, :, loc[0]-1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax2.set_title('Noncontrast')
    #fig.colorbar(img1, ax=ax1)

    ax3.imshow(nc_image[:, :, loc[0]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax3.imshow(nc_mask[:, :, loc[0]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax3.set_title('Noncontrast')
    #fig.colorbar(img1, ax=ax1)

    ax4.imshow(nc_image[:, :, loc[0]+1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax4.imshow(nc_mask[:, :, loc[0]+1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax4.set_title('Noncontrast')
    #fig.colorbar(img1, ax=ax1)

    ax5.imshow(nc_image[:, :, loc[0]+2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax5.imshow(nc_mask[:, :, loc[0]+2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax5.set_title('Noncontrast')
    #fig.colorbar(img1, ax=ax1)

    ax6.imshow(phs_image_reg[:, :, loc[1]-2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax6.imshow(phs_mask_reg[:, :, loc[1]-2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax6.set_title('Phase (After reg)')
    #fig.colorbar(img1, ax=ax1)

    ax7.imshow(phs_image_reg[:, :, loc[1]-1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax7.imshow(phs_mask_reg[:, :, loc[1]-1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax7.set_title('Phase (After reg)')
    #fig.colorbar(img1, ax=ax1)

    ax8.imshow(phs_image_reg[:, :, loc[1]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax8.imshow(phs_mask_reg[:, :, loc[1]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax8.set_title('Phase (After reg)')
    #fig.colorbar(img1, ax=ax1)

    ax9.imshow(phs_image_reg[:, :, loc[1]+1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax9.imshow(phs_mask_reg[:, :, loc[1]+1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax9.set_title('Phase (After reg)')
    #fig.colorbar(img1, ax=ax1)

    ax10.imshow(phs_image_reg[:, :, loc[1]+2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax10.imshow(phs_mask_reg[:, :, loc[1]+2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax10.set_title('Phase (After reg)')
    #fig.colorbar(img1, ax=ax1)

    ax11.imshow(nc_image[:, :, loc[0]-2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax11.imshow(nc_mask[:, :, loc[0]-2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax11.imshow(phs_image_reg[:, :, loc[1]-2], cmap="jet", interpolation='none', vmin=-200, vmax=800, alpha=0.2)
    ax11.imshow(phs_mask_reg[:, :, loc[1]-2], cmap="jet", interpolation='none', alpha=0.5)
    ax11.set_title('Phase (After reg) Overlaid on Noncontrast')

    ax12.imshow(nc_image[:, :, loc[0]-1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax12.imshow(nc_mask[:, :, loc[0]-1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax12.imshow(phs_image_reg[:, :, loc[1]-1], cmap="jet", interpolation='none', vmin=-200, vmax=800, alpha=0.2)
    ax12.imshow(phs_mask_reg[:, :, loc[1]-1], cmap="jet", interpolation='none', alpha=0.5)
    ax12.set_title('Phase (After reg) Overlaid on Noncontrast')

    ax13.imshow(nc_image[:, :, loc[0]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax13.imshow(nc_mask[:, :, loc[0]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax13.imshow(phs_image_reg[:, :, loc[1]], cmap="jet", interpolation='none', vmin=-200, vmax=800, alpha=0.2)
    ax13.imshow(phs_mask_reg[:, :, loc[1]], cmap="jet", interpolation='none', alpha=0.5)
    ax13.set_title('Phase (After reg) Overlaid on Noncontrast')

    ax14.imshow(nc_image[:, :, loc[0]+1], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax14.imshow(nc_mask[:, :, loc[0]+1], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax14.imshow(phs_image_reg[:, :, loc[1]+1], cmap="jet", interpolation='none', vmin=-200, vmax=800, alpha=0.2)
    ax14.imshow(phs_mask_reg[:, :, loc[1]+1], cmap="jet", interpolation='none', alpha=0.5)
    ax14.set_title('Phase (After reg) Overlaid on Noncontrast')

    ax15.imshow(nc_image[:, :, loc[0]+2], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax15.imshow(nc_mask[:, :, loc[0]+2], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax15.imshow(phs_image_reg[:, :, loc[1]+2], cmap="jet", interpolation='none', vmin=-200, vmax=800, alpha=0.2)
    ax15.imshow(phs_mask_reg[:, :, loc[1]+2], cmap="jet", interpolation='none', alpha=0.5)
    ax15.set_title('Phase (After reg) Overlaid on Noncontrast')

    plt.show()

    if savepath:
        fig.savefig(savepath)


def visualize_reg_mask(phase_path, mask_path, slice_nums):
    # visualize mask on phase image given four slice numbers in form [slice1, slice2, slice3, slice4]

    nii_img = nib.load(phase_path)
    phase_image = nii_img.get_fdata()

    nii_img = nib.load(mask_path)
    mask = nii_img.get_fdata()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(20,10), ncols=4) 

    ax1.imshow(phase_image[:, :, slice_nums[0]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax1.imshow(mask[:, :, slice_nums[0]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax1.set_title('Axial Slice 1')

    ax2.imshow(phase_image[:, :, slice_nums[1]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax2.imshow(mask[:, :, slice_nums[1]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax2.set_title('Axial Slice 2')

    ax3.imshow(phase_image[:, :, slice_nums[2]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax3.imshow(mask[:, :, slice_nums[2]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax3.set_title('Axial Slice 3')

    ax4.imshow(phase_image[:, :, slice_nums[3]], cmap="Greys_r", interpolation='none', vmin=-200, vmax=800)
    ax4.imshow(mask[:, :, slice_nums[3]], cmap="Greys_r", interpolation='none', alpha=0.5)
    ax4.set_title('Axial Slice 4')

    plt.show()


def view_reg_path_oneslice(path_1, path_2_reg, path_2, loc, **kwargs):
        savepath = kwargs.get('savepath', None)

        nii_img = nib.load(path_1)
        header = nii_img.header
        print(header.get_data_shape())
        image1 = nii_img.get_fdata()

        nii_img = nib.load(path_2_reg)
        header = nii_img.header
        print(header.get_data_shape())
        image2_reg = nii_img.get_fdata()

        nii_img = nib.load(path_2)
        header = nii_img.header
        print(header.get_data_shape())
        image2 = nii_img.get_fdata()

        view_registration_oneslice(image1, image2_reg, image2, loc, savepath=savepath)


def view_registration_oneslice(image1, image2_reg, image2, loc, **kwargs):
    
    savepath = kwargs.get('savepath', None)

    if not loc:
        mid1 = int(image1.shape[2]/2) #plot middle slice
        loc = [mid1, mid1]

    fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(figsize=(5,20), ncols=1, nrows=4)  


    img1 = ax1.imshow(image1[:, :, loc[0]], cmap="Greys_r", vmin=-200, vmax=800)
    #ax1.set_title('Pre-contrast')
    ax1.set_xticks([])
    ax1.set_yticks([])

    img2 = ax2.imshow(image2[:, :, loc[1]], cmap="Greys_r", vmin=-200, vmax=800)
    #ax2.set_title('Post-Contrast, Pre-registration')
    ax2.set_xticks([])
    ax2.set_yticks([])

    img3 = ax3.imshow(image2_reg[:, :, loc[1]], cmap="Greys_r", vmin=-200, vmax=800)
    #ax3.set_title('Post-Contrast, Post-registration')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4.imshow(image1[:, :, loc[0]], cmap="Greys_r", vmin=-200, vmax=800)
    ax4.imshow(image2_reg[:, :, loc[1]], cmap="jet", vmin=-200, vmax=800, alpha=0.2)
    #ax4.set_title('Pre-Contrast with Post-Contrast \n Post-Reg Overlaid')
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath)
