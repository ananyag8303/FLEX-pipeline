# FLEX-pipeline

Repository created to expand disc galaxies from the CEERS dataset using Fourier - Laguerre Polynomials. 

![Alt text](Paper_Grid.png)

FITS File Format

f444w({field_number}).fits, f410m({field_number}).fits, f356w({field_number}).fits, f277w({field_number}).fits, f200w({field_number}).fits, f115w({field_number}).fits, f125w({field_number}).fits, f160w({field_number}).fits, f814w({field_number}).fits, f606w({field_number}).fits

Note: For EGS23205, Field number is 3. 

Other things to modify:

- In the Example_Run.ipynb notebook, one must change the condiotion set for FITS file based on the FITS file present in the directory. 'fits_files = [f'f356w({i}).fits' for i in range(3, 4)]'. The same thing must be done for Example_Run-Error.ipynb
- The CANDELS catalogue for this pipeline can be found here: https://archive.stsci.edu/hlsp/candels/egs-catalogs . The file I used is called: hlsp_candels_hst_wfc3_egs_multi_v1_mass-cat.csv.txt 
- CEERS (JWST + HST) FITS files can be found here: https://ceers.github.io/dr05.html
