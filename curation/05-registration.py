
# script based on 05-registration-round2.ipynb to register and save for QC later a batch of exams
# for now just do affine registration without mask

# %%
import numpy as np
from dipy.viz import regtools
from dipy.align import affine_registration
from dipy.align.imaffine import AffineMap
import nibabel as nib
import os
import matplotlib.pyplot as plt
from curation.utils import view_registration, view_registration_with_mask, find_midz, get_phase, get_mask_fname
from tqdm import tqdm

# %%
data_path = "/data/larson2/RCC_dl/data2022/"
qc_path = '/working/larson2/ssahin/rcc_dl/QC'
pid_list = ["LI7VGJbXvc"]   

for pid in tqdm(pid_list):

    print('--------------------'+pid+'--------------------')
    phases = get_phase(os.path.join(data_path,pid))
    print(phases)
    #if os.path.exists(os.path.join(data_path,pid,'noncon_cropped.nii.gz')):
    #    crop_suff = "_cropped"
    #else:
    #    crop_suff = ""
    crop_suff = ""
    static_fname = 'noncon' + crop_suff + '.nii.gz'
    static_mask_fname = get_mask_fname(os.path.join(data_path,pid), "noncon", suffix=crop_suff)
    nc_mask_bool = os.path.exists(os.path.join(data_path,pid,static_mask_fname))
        
    static = nib.load(os.path.join(data_path,pid,static_fname))
    static_img = static.get_fdata()
    
    if nc_mask_bool:
        static_mask = nib.load(os.path.join(data_path,pid,static_mask_fname))
        static_mask_img = static_mask.get_fdata()
        minz, midz, maxz =find_midz(static_mask_img)

    if not os.path.exists(os.path.join(qc_path,pid)):
        os.makedirs(os.path.join(qc_path,pid))

    #phase = phases[0]
    for phase in phases:

        print('--------------------'+phase+'--------------------')
        moving_fname = phase + crop_suff + '.nii.gz'
        moving_mask_fname = get_mask_fname(os.path.join(data_path,pid), phase, suffix=crop_suff)
        moving = nib.load(os.path.join(data_path,pid,moving_fname))
        moving_img = moving.get_fdata()

        phs_mask_bool = os.path.exists(os.path.join(data_path,pid,moving_mask_fname))
        if phs_mask_bool:
            moving_mask = nib.load(os.path.join(data_path,pid,moving_mask_fname))    
            moving_mask_img = moving_mask.get_fdata()
            if not nc_mask_bool:
                minz, midz, maxz =find_midz(moving_mask_img)

    # %%
        #Plot Initial
        identity = np.eye(4)
        affine_map = AffineMap(identity,
                            static_img.shape, static.affine,
                            moving_img.shape, moving.affine)
        resampled = affine_map.transform(moving_img)

        regtools.overlay_slices(static_img, resampled, None, 0,
                            "Static", "Moving-resampled", os.path.join(qc_path,pid,(phase+'_'+'initial_0.png')))
        regtools.overlay_slices(static_img, resampled, None, 1,
                            "Static", "Moving-resampled", os.path.join(qc_path,pid,(phase+'_'+'initial_1.png')))
        regtools.overlay_slices(static_img, resampled, None, 2,
                            "Static", "Moving-resampled", os.path.join(qc_path,pid,(phase+'_'+'initial_2.png')))

        # %%
        # AFFINE REGISTRATION

        pipeline = ["center_of_mass", "translation", "rigid", "affine"]
        level_iters = [10000, 1000, 100] #number of iterations at three diff resolutions 10000 at the coarsest, 100 at finest
        sigmas = [3.0, 1.0, 0.0] #sigma values for gaussian pyramid
        factors = [4, 2, 1] #controls res of registration i.e. coarsest reg here would done at (nx//4, ny//4, nz//4) where (nx,ny,nz) is OG

        xformed_img, reg_affine = affine_registration(
            moving_img,
            static_img,
            moving_affine=moving.affine,
            static_affine=static.affine,
            nbins=32,
            metric='MI',
            pipeline=pipeline,
            level_iters=level_iters,
            sigmas=sigmas,
            factors=factors)
            #static_mask = static_mask_img,
            #moving_mask = moving_mask_img)

        # also register mask with same transform
        if phs_mask_bool:
            if nc_mask_bool:
                affine_map = AffineMap(reg_affine,
                            static_mask_img.shape, static_mask.affine,
                            moving_mask_img.shape, moving_mask.affine)
            else:
                affine_map = AffineMap(reg_affine,
                            static_img.shape, static.affine,
                            moving_mask_img.shape, moving_mask.affine)
            moving_mask_img_xformed = affine_map.transform(moving_mask_img)

        # %%
        # QC pt 1

        regtools.overlay_slices(static_img, xformed_img, None, 0,
                                "Static", "Transformed", os.path.join(qc_path,pid,(phase+'_xformed_affine_0.png')))
        regtools.overlay_slices(static_img, xformed_img, None, 1,
                                "Static", "Transformed", os.path.join(qc_path,pid,(phase+'_xformed_affine_1.png')))
        regtools.overlay_slices(static_img, xformed_img, None, 2,
                                "Static", "Transformed", os.path.join(qc_path,pid,(phase+'_xformed_affine_2.png')))

        # QC pt 2

        try:
            view_registration(static_img, xformed_img, [minz, minz], savepath=os.path.join(qc_path,pid,(phase+'_minz.png')))
            view_registration(static_img, xformed_img, [midz, midz], savepath=os.path.join(qc_path,pid,(phase+'_midz.png')))
            view_registration(static_img, xformed_img, [maxz, maxz], savepath=os.path.join(qc_path,pid,(phase+'_maxz.png')))
        except:
            pass
        
        if phs_mask_bool and nc_mask_bool:
            try:
                view_registration_with_mask(static_img, xformed_img, static_mask_img, moving_mask_img_xformed, [minz, minz], savepath=os.path.join(qc_path,pid,(phase+'_minz_wmask.png')))
                view_registration_with_mask(static_img, xformed_img, static_mask_img, moving_mask_img_xformed,  [midz, midz], savepath=os.path.join(qc_path,pid,(phase+'_midz_wmask.png')))
                view_registration_with_mask(static_img, xformed_img, static_mask_img, moving_mask_img_xformed,  [maxz, maxz], savepath=os.path.join(qc_path,pid,(phase+'_maxz_wmask.png')))
            except:
                pass

        # %%
        print(static_img.shape)
        print(moving_img.shape)
        print(xformed_img.shape)

        if phs_mask_bool and nc_mask_bool:
            print(static_mask_img.shape)
            print(moving_mask_img.shape)
            print(moving_mask_img_xformed.shape)

        # %%
        # save finalized images as niftis
        new_affine = moving.affine.dot(reg_affine)
        xformed = nib.Nifti1Image(xformed_img, affine=new_affine, header=moving.header)
        xformed_path = os.path.join(data_path,pid,(phase + "_reg.nii.gz"))
        nib.save(xformed, xformed_path)

        if phs_mask_bool:
            new_affine_mask = moving_mask.affine.dot(reg_affine)
            xformed_mask = nib.Nifti1Image(moving_mask_img_xformed, affine=new_affine_mask, header=moving_mask.header) 
            
            if crop_suff:
                xformed_mask_path = os.path.join(data_path,pid, (moving_mask_fname[:-15] + "_reg.nii.gz"))
            else:
                xformed_mask_path = os.path.join(data_path,pid, (moving_mask_fname[:-7] + "_reg.nii.gz"))   

            nib.save(xformed_mask, xformed_mask_path)
        # %%
