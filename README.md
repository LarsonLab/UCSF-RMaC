# UCSF RMaC: UCSF Renal Mass CT Dataset

![Logo](logo.png)

Scripts are numbered 01-07 to indicate each step of curation process

Sample conda environment can be found in `environment.yml`

`utils.py` -- contains utility functions for all steps

## File Structure of Dataset

Data folder includes HDF5 files named by patient ID as well as the key csv which labels the phases that exist in each file and whether or not they are registered.

Within phase_reg_key.csv: 
- 0 = no volume
- 1 = volume exists but is not registered to the unenhanced (noncon) volume
- 2 = volume exists and is registered to the unenhanced (noncon) volume

```
.
├── 08FBroxzI6.hdf5
├── 0A87Rq5Hkl.hdf5
├── 0ByGP3oWJi.hdf5
├── 0cb2z7Hao2.hdf5
...
├── phase_reg_key.csv
...
├── Zu1bNdA2od.hdf5
├── ZYUz7t5hOn.hdf5
└── Zz99Ji2swU.hdf5
```
