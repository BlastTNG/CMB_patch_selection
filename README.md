# CMB_patch_selection
Simulations to select the CMB patches to be observed.

The code "blast_VF_last_version.py" generates the maps and the quality parameters of the patches, in the Blast's visibility area 10 hours.
The metric can be generated both considering patches of size 2x2 degrees, both 5x5 degrees (parameter "patch_size").
The map resolution is equal to 1.7 arcmin or to 30 arcmin (PySM parameter: nside = 2048 and 128 respectively).

LIMITS:
- At the moment the code only works with square patches
- At the moment it is considered a single partition of the visibility area (i.e. a binning of the intensities)

IMPROVEMENTS TO BE MADE:

- To made a partition of the visibility area even with rectangular patches (3x5 degrees)
- To consider different partitions, that can be obtained by translating the centers of the patches into longitude and latitude

