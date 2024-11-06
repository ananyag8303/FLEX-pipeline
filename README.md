# flex-pipeline

FITS File Format

f444w({field_number}).fits, f410m({field_number}).fits, f356w({field_number}).fits, f277w({field_number}).fits, f200w({field_number}).fits, f115w({field_number}).fits, f125w({field_number}).fits, f160w({field_number}).fits, f814w({field_number}).fits, f606w({field_number}).fits

Note: For 23205, Field number is 3. 

Other things to modify:

- In the Example_Run.ipynb notebook, one must change the condiotion set for FITS file based on the FITS file present in the directory. 'fits_files = [f'f356w({i}).fits' for i in range(3, 4)]'. The same thing must be done for Example_Run-Error.ipynb
