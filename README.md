# FLEX-pipeline

Repository created to expand disc galaxies from the CEERS dataset using Fourier - Laguerre Polynomials. This pipeline has currently been set up to expand disc galaxies in 6 JWST filters (F444W, F4120M, F356W, F277W, F200W and F115W) and 4 HST filters (F125W, F160W, F606W and F814W). The below image demonstrates the reconstructed expansion of EGS23205 in all ten filters (bottom panel) as compared to the original surface density. 

![Alt text](Paper_Grid.png)

The following is a summary of the files in this repository:

1. `Accounting.ipynb`: Summary of disc galaxies used from Ferreira et al (2023) database available [here.](https://github.com/astroferreira/CEERS_EPOCHS_MORPHO/tree/main)
2. `Code_Overview.ipynb`: Notebook detailing the structure of ***FLEX***
3. `Math_Overview.ipynb`: Notebook detailing the mathematics behind ***FLEX***. This is based on initial work presented in [here.](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.5408W/abstract)
4. `FLEX.py`: File containing the main classes that produced coefficients as described in `Code_Overview.ipynb`. 
5. `FLEX_Error.py`: File that runs and creates 100 image realisations of a galaxy and finds respective coefficients. 
6. `Example_Run.ipynb`: Notebook to run the main ***FLEX*** pipeline from `FLEX.py`
7. `Example_Run-Error.ipynb`: Notebook to run pipeline from `FLEX_Error.py`
8. `summary_values.csv`: CSV file summarising the following information for each galaxy: `RA`, `Dec`, `Redshift`, `Stellar Mass`, `SFR`, `Asymmetry` in all 10 filters. Note `RA`, `Dec`, `Redshift`, `Stellar Mass` and `SFR` values have been obtained from the CANDELS catalogue. 

The below table displays the RA and Dec bounds for all fields imaged through CEERS. This is useful when identifying which FITS files to download. For reference, EGS23205 lies in Field Number 3. 

| Field Number | RA Min             | RA Max            | Dec Min            | Dec Max            |
|--------------|--------------------|-------------------|--------------------|--------------------|
|      1       | 214.91059215475673 | 215.056273520994  | 52.93102145859292  | 53.02419052250401  |
|      2       | 214.81753938116827 | 214.98386177597308| 52.85612163208139  | 52.960766480428575 |
|      3       | 214.73106352250988 | 214.87525584189984| 52.80452870656189  | 52.897129917558054 |
|      4       | 214.68669748331905 | 214.83057385761245| 52.72184280085484  | 52.81446001663214  |
|      5       | 214.85214976038571 | 215.0234550434128 | 52.8417542533767   | 52.94983314050119  |
|      6       | 214.78327910492473 | 214.92745941240094| 52.786128133664604 | 52.878710410117094 |
|      7       | 215.0171118006846  | 215.1887959605763 | 52.90148470657302  | 53.00951454702593  |
|      8       | 214.92465941168712 | 215.0960105325861 | 52.83745618649953  | 52.94551386733736  |
|      9       | 214.8318590344492  | 215.00288241694562| 52.77475611977953  | 52.88284114363774  |
|      10      | 214.75687550159984 | 214.90080224951433| 52.717112379883616 | 52.80970419115057  |

This repository accepts file names in the following format for easier handling:

- f444w(`field_number`).fits
- f410m(`field_number`).fits
- f356w(`field_number`).fits
- f277w(`field_number`).fits
- f200w(`field_number`).fits
- f115w(`field_number`).fits
- f125w(`field_number`).fits
- f160w(`field_number`).fits
- f814w(`field_number`).fits
- f606w(`field_number`).fits

Besides this, please keep a note of the following points before using ***FLEX***

- CEERS JWST and archival HST FITS files can be found [here.](https://ceers.github.io/dr05.html) While this pipeline has been set up to expand disc galaxies from the CEERS dataset, it can be adapted to other surveys like PRIMER and JADES too. 
- The CANDELS catalogue for this pipeline can be found [here.](https://archive.stsci.edu/hlsp/candels/egs-catalogs) I refered to `hlsp_candels_hst_wfc3_egs_multi_v1_mass-cat.txt` to get RA and Dec values. (Note: Upon downloading, the filename changes to `hlsp_candels_hst_wfc3_egs_multi_v1_mass-cat.csv.txt`)
- In the Example_Run.ipynb notebook, one must change the condiotion set for searching through FITS files bounds based on the FITS file present in the directory. The line to change is `fits_files = [f'f356w({i}).fits' for i in range(3, 4)]`. Currently, this only searches through FITS files related to Field Number 3. The same thing must be done for `Example_Run-Error.ipynb`

If you use this ***FLEX*** as part of your research, we kindly ask you to reference it as:

`@ARTICLE{Ganapathy_etal(2023),\
       author = {{Ganapathy}, Ananya and {Petersen}, Michael S. and {Yaaqib}, Rashid and {Filion}, Carrie},\
        title = "{Disc asymmetry characterisation in JWST-observed galaxies at 1 < z < 4}",\
      journal = {arXiv e-prints},\
     keywords = {Astrophysics - Astrophysics of Galaxies},\
         year = 2024,\
        month = nov,\
          eid = {arXiv:2411.11972},\
        pages = {arXiv:2411.11972},\
          doi = {10.48550/arXiv.2411.11972},\
archivePrefix = {arXiv},\
       eprint = {2411.11972},\
 primaryClass = {astro-ph.GA},\
}`