## looped script, create pkl with pids in 06_to_hdf5.ipynb

import nibabel as nib
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import pickle
from utils import get_best_mask

# load in saved pids list and dataframe
with open('./pids_reg_label_07172024.txt', 'rb') as f:
    updated_pids = pickle.load(f)
df_labels_pids = pd.read_pickle("./df_labels_pid_07172024.pkl") 
#df_labels_pids = pd.read_csv("./pids_registered_stacked_7_17_24.csv", header=None, names=["PID"])   

save_dir ="/data/larson2/RCC_dl/hdf5_dir/"
data_path = "/data/larson2/RCC_dl/data2022/"
phase_fname = {"noncon": "noncon_cropped.nii.gz","arterial": "arterial_reg.nii.gz", "delay": "delay_reg.nii.gz", "portven": "portven_reg.nii.gz"}

updated_pids = ["LI7VGJbXvc"] #to test
istart = 0
#iend = 260

for pid in tqdm(updated_pids[istart:]):
    print("PID:", pid)
    # create and open hdf5 file
    h5_fname = os.path.join(save_dir,pid + ".hdf5")
    f = h5py.File(h5_fname, "w")

    # add metadata
    f.attrs["PID"] = pid
    f.attrs["tumor_type"] = df_labels_pids.loc[df_labels_pids["Anon MRN"] == pid]["tumor_type"].values[0]
    f.attrs["pathology"] = df_labels_pids.loc[df_labels_pids["Anon MRN"] == pid]["pathology"].values[0]
    f.attrs["pathology_grade"] = df_labels_pids.loc[df_labels_pids["Anon MRN"] == pid]["grade"].values[0]

    # add all registered phase images to hdf5
    for phase in phase_fname:
        if os.path.exists(os.path.join(data_path,pid,phase_fname[phase])):
            # load in nifti
            image = nib.load(os.path.join(data_path,pid,phase_fname[phase]))
            image_np = image.get_fdata()

            # add image to hdf5
            f.create_dataset(phase, data=image_np)

            # add pixel spacing
            f.attrs[phase+"_pixdim"] = image.header["pixdim"][1:4]

            masktag = "cropped"

        if phase =="noncon": ## need to catch not cropped noncon -- maybe copy them here
            if not os.path.exists(os.path.join(data_path,pid,phase_fname[phase])):
                print("noncon isnt cropped")

                # load in nifti
                image = nib.load(os.path.join(data_path,pid,"noncon.nii.gz"))
                image_np = image.get_fdata()

                # add image to hdf5
                f.create_dataset(phase, data=image_np)

                # add pixel spacing
                f.attrs[phase+"_pixdim"] = image.header["pixdim"][1:4]

                masktag = "not_cropped"

    # pick best mask and convert to hdf5
    mask_fname = get_best_mask(os.path.join(data_path,pid), masktag)
    # load in nifti
    image = nib.load(os.path.join(data_path,pid,mask_fname))
    mask_np = image.get_fdata()
    # threshold mask
    mask_np[mask_np < 0.5] = 0
    mask_np[mask_np >= 0.5] = 1

    # save image as hdf5
    f.create_dataset("mask", data=mask_np)

    # add mask pixel spacing
    f.attrs["mask_pixdim"] = image.header["pixdim"][1:4]

    f.close()