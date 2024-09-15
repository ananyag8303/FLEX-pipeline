# Import necessary Python libraries and packages

import chardet
import numpy as np
import time
import h5py
import warnings
import csv
import os

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from scipy.special import eval_genlaguerre
from scipy.optimize import curve_fit
from scipy import stats

class GalaxyDataExtractor:
    """
    A class to extract and process galaxy data from FITS files and EGS CANDELS catalogue.

    Parameters:
    ----------
    fits_files : list of str
        List of paths to FITS files containing galaxy data.
    filename : str
        Path to the CANDELS catalogue.
    galaxy_ids : list of int
        List of galaxy IDs to process.

    Attributes:
    ----------
    fits_files : list of str
        List of paths to FITS files containing galaxy data.
    filename : str
        Path to the EGS CANDELS catalogue.
    galaxy_ids : list of int
        List of user inputted galaxy IDs to process.
    wcs_objects : dict
        Dictionary mapping each FITS file to its WCS object.
    """

    def __init__(self, fits_files, filename, galaxy_ids):
        """
        Initialize GalaxyDataExtractor class with FITS files, CANDELS catalogue, and galaxy IDs.

        Here, specific warnings related to FITS files and runtime warnings are supressed.

        Parameters:
        ----------
        fits_files : list of str
            List of paths to FITS files containing galaxy data.
        filename : str
            Path to the CANDELS catalogue.
        galaxy_ids : list of int
            List of galaxy IDs to process.
        """
        # Suppress specific warnings as they crowd the output panel and don't have any purpose
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in .*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in .*')
        
        # Set of FITS files to search through 
        self.fits_files = fits_files
        
        # Filename here is the CANDELS catalogue 
        self.filename = filename
        
        # User created list of galaxy IDs
        self.galaxy_ids = galaxy_ids
        
        # WCS initialisation
        self.wcs_objects = {file: WCS(fits.open(file)[1].header) for file in fits_files}

    def detect_encoding(self, filename):
        """
        Detect the encoding of the CANDELS catalogue file.

        Parameters:
        ----------
        filename : str
            Path to the CANDELS catalogue.

        Returns:
        -------
        str
            Encoding detected from the CANDELS catalogue.
        """
        # Open file (binary) in read mode
        with open(filename, 'rb') as file:
            
            # Read in the first 10,000 bytes
            raw_data = file.read(10000)
            
        # Detect encoding from analyzed bytes
        result = chardet.detect(raw_data)
        
        # Return the resulting encoding
        return result['encoding']

    def get_ra_dec(self, filename, galaxy_id):
        """
        Retrieve RA (Right Ascension) and Dec (Declination) values for a specific galaxy ID
        from the CANDELS catalogue.

        Parameters:
        ----------
        filename : str
            Path to the CANDELS catalogue.
        galaxy_id : int
            ID of the galaxy to retrieve RA and Dec.

        Returns:
        -------
        tuple of float or None
            RA and Dec values of the galaxy if found, otherwise (None, None).
        """
        # Use above function to determine encoding of catalogue
        encoding = self.detect_encoding(filename)
        
        # Try and except block to handle possible encoding errors
        try:
            # Open file using encoding determined above
            with open(filename, 'r', encoding=encoding) as file:
                
                # For loop to go through each line in file
                for line in file:
                    data = line.strip().split()
                    
                    # If loop to skip beginning header lines and move on to actual data
                    if len(data) < 3:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue
                        
                    # If user entered ID matches ID in catalogue
                    if data[0] == str(galaxy_id):
                        
                        # Try and except block to handle cases when wrong ID could have been entered by user
                        try:
                            # Extract RA and Dec stored in 2nd and 3rd column of txt file respectively
                            ra = float(data[1])
                            dec = float(data[2])
                            return ra, dec
                        
                        except (IndexError, ValueError) as e:
                            print(f"Error extracting RA/Dec values: {str(e)}")
                            continue
        
        except UnicodeDecodeError as e:
            print(f"Error decoding file with encoding {encoding}: {str(e)}")
        return None, None

    def sersic_profile(self, r, I0, Reff, n):
        """
        Calculate the Sersic profile for galaxy luminosity distribution.

        Parameters:
        ----------
        r : float or array_like
            Radial distance from the galactic center.
        I0 : float
            Central surface brightness.
        Reff : float
            Effective galaxy radius.
        n : float
            Sersic index number.

        Returns:
        -------
        float or array_like
            Luminosity distribution at given radial distance from galactic centre.
        """
        return I0 * np.exp(- (r / Reff)**(1/n))

    def determine_galaxy_radius(self, image_data, n_guess=1.0):
        """
        Determine the main galaxy radius beyond which background galaxies are masked.

        Parameters:
        ----------
        image_data : ndarray
            Image data of the galaxy.
        n_guess : float, optional
            Initial guess for the Sersic index (default is 1.0 as they approximates to an exponential profile).

        Returns:
        -------
        float
            Radius of the main galaxy beyond which background galaxies are masked.
        """
        
        # Calculate the cutout center and image dimensions
        xdim, ydim = image_data.shape
        x_center, y_center = xdim / 2, ydim / 2
        
        # Define a radial distance array from the galactic center
        x_indices, y_indices = np.indices(image_data.shape)
        radial_distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        
        # Flatten image data and radial distance array created above
        intensity_values = image_data.flatten()
        radial_distances_flat = radial_distances.flatten()
        
        # Remove NaN values as they can giv rise to an error
        mask = ~np.isnan(intensity_values)
        intensity_values = intensity_values[mask]
        radial_distances_flat = radial_distances_flat[mask]
        
        # Sort intensity values by radial distance
        sorted_indices = np.argsort(radial_distances_flat)
        sorted_distances = radial_distances_flat[sorted_indices]
        sorted_intensities = intensity_values[sorted_indices]
        
        # If loop to deal with cases where there aren't enough points for scipy.optimize curvefit method to work on
        if len(sorted_distances) < 10:
            print("Not enough data points for fitting.")
            return np.nan
        
        # Try and except block to handle possible errors in Sersic Profile fitting
        try:
            # Perform scipy curve fitting using scipy.optimize. Disk galaxies here have exponential profile ie n ~ 1
            popt, _ = curve_fit(self.sersic_profile, sorted_distances, sorted_intensities, p0=(1.0, sorted_distances[-1], n_guess), maxfev=10000)
            
            # Assign values to the result got above
            I0, Reff, n = popt
            
            # Define the galaxy radius as a multiple of effective radius
            galaxy_radius = 6 * Reff 
            
            return galaxy_radius
        
        except Exception as e:
            print(f"Error fitting Sersic profile: {str(e)}")
            return np.nan

    def get_fits_data(self, filename, rounded_x, rounded_y, galaxy_radius=None):
        """
        Obtain image array of the main galaxy while masking out background galaxies.

        Parameters:
        ----------
        filename : str
            Path to the FITS file containing galaxy image data.
        rounded_x : int
            Rounded x-coordinate of the galaxy.
        rounded_y : int
            Rounded y-coordinate of the galaxy.
        galaxy_radius : float, optional
            Radius of the galaxy beyond which background galaxies are masked (default is None).

        Returns:
        -------
        ndarray
            Cropped image data of the galaxy.
        ndarray
            Cropped uncertainty data of the galaxy.
        """
        # Open FITS file
        with fits.open(filename) as hdu:
            
            # Extract image data and its related uncertainty
            image_data = hdu[1].data
            image_uncertainty = hdu[3].data
            
            # Define pixel cutout bounds
            xmin, xmax = rounded_x - 55, rounded_x + 55
            ymin, ymax = rounded_y - 55, rounded_y + 55
            
            # Crop data and uncertainty using above created pixel box dimensions
            cropped_data = image_data.T[xmin:xmax, ymin:ymax]
            cropped_uncertainty = image_uncertainty.T[xmin:xmax, ymin:ymax]
            
            # Transpose image data and uncertainty
            cropped_data = cropped_data.T
            cropped_uncertainty = cropped_uncertainty.T
            
            # If galaxy radius isn't fixed find it. This will be done for F444W and kept fixed for all other filters
            if galaxy_radius is None:
                galaxy_radius = self.determine_galaxy_radius(cropped_data)
            
            # Calculate radial distance from galactic center (rounded_x, rounded_y) which is RA and Dec value from CANDELS
            x_indices, y_indices = np.indices(cropped_data.shape)
            distance_from_center = np.sqrt((x_indices - (rounded_x - xmin))**2 + (y_indices - (rounded_y - ymin))**2)
            
            # Mask high pixel intensity spots only outside a certain radius
            mask_galaxy = distance_from_center <= galaxy_radius
            
            # Define a threshold for background galaxy masking using sigma clipping
            metric_threshold = np.median(cropped_data[mask_galaxy]) + 3 * stats.median_abs_deviation(cropped_data[mask_galaxy])
            
            # Define threshold where if the pixel value (outside the galaxy radius) exceeds metric_threshold it is blanked
            mask_background = cropped_data > metric_threshold
            
            # Combine galaxy and background masks to blank out regions. '~' reverses the conditions of mask_galaxy
            mask_to_blank = mask_background & ~mask_galaxy
            
            # Apply NaNs to blank out regions in cropped_data and cropped_uncertainty
            cropped_data[mask_to_blank] = np.nan
            cropped_uncertainty[mask_to_blank] = np.nan

        return cropped_data, cropped_uncertainty

    def create_full_image(self, cropped_data, cropped_uncertainty):
        """
        Create a full image grid with pixel coordinates, image data, and uncertainty.

        Parameters:
        ----------
        cropped_data : ndarray
            Cropped image data of the galaxy.
        cropped_uncertainty : ndarray
            Cropped uncertainty data of the galaxy.

        Returns:
        -------
        ndarray
            Full image array with pixel coordinates, image data, and uncertainty.
        """
        # Find x and y dimension of cropped image
        xdim, ydim = cropped_data.shape
        
        # Use this information to create a grid for plotting
        xpixels = np.linspace(-xdim / 2, xdim / 2, xdim)
        ypixels = np.linspace(-ydim / 2, ydim / 2, ydim)
        xarr, yarr = np.meshgrid(xpixels, ypixels, indexing='ij')
        
        # Stack all pixel information (4 arrays per image)
        full_image = np.stack([xarr, yarr, cropped_data, cropped_uncertainty])
        
        return full_image

    def process_galaxies(self):
        """
        Process each galaxy in the list of galaxy IDs, extract relevant data from FITS files and CANDELS catalogue,
        and save the processed data into HDF5 files organized by different filters.

        This method iterates over each galaxy ID provided during initialization. For each galaxy ID:
        - It retrieves the RA and Dec coordinates from the CANDELS catalogue.
        - Finds the corresponding FITS file where the galaxy is located based on its RA and Dec values.
        - Determines the rounded pixel coordinates within the FITS image.
        - Calculates the galaxy radius using the F444W filter data.
        - Extracts image data and uncertainty arrays from multiple filters centered on the galaxy,
          masking out background galaxies beyond the determined radius.
        - Creates full image arrays including pixel coordinates, image data, and uncertainty.
        - Saves each image array into an HDF5 file named 'testing_{galaxy_id:05d}.hdf5', organized into groups
          corresponding to each filter.

        The HDF5 files contain structured data suitable for further analysis and visualization of galaxy images.

        Note:
        - If the galaxy ID is not found in the CANDELS catalogue or its coordinates do not correspond to any FITS file,
          a message is printed and the process moves to the next galaxy ID.
        - Errors encountered during data extraction or file handling are caught and logged.

        Raises:
        ------
        OSError
            If there is an issue creating or writing to the HDF5 files.
        """
        # For loop to go through each galaxy in user created list
        for galaxy_id in self.galaxy_ids:

            # Print Galaxy ID currently being analysed
            print(f"Processing galaxy ID: {galaxy_id}")

            # Get RA and Dec values using above defined function
            RA, Dec = self.get_ra_dec(self.filename, galaxy_id)

            # If no RA and Dec is found report that it doesn't exist in given FITS fields. Move onto next galaxy
            if RA is None or Dec is None:
                print(f"Galaxy ID {galaxy_id} not found in the file.")
                continue

            valid_x, valid_y, valid_file = None, None, None
            # For loop to go through each FITS file 
            for file in self.fits_files:

                # Open FITS file
                with fits.open(file) as data:

                    # Create a WCS object
                    wcs = WCS(data[1].header)

                    # Convert world coordinates to pixel coordinates
                    x, y = wcs.world_to_pixel_values(RA, Dec)

                    # Check if pixel coordinates are within bounds of the image
                    if 0 <= x < data[1].data.shape[1] and 0 <= y < data[1].data.shape[0]:
                        valid_x, valid_y, valid_file = x, y, file
                        # Break loop when valid file and pixel values are found 
                        break

            # Convert RA and Dec values to pixel coordinates using 'if' loop
            if valid_x is not None and valid_y is not None:
                
                rounded_x, rounded_y = np.round(valid_x).astype(int), np.round(valid_y).astype(int)
                
                # Print FITS file in which galaxy exists in
                print(f"RA and Dec are within the bounds of: {valid_file}")
                
                # Print rounded pixel coordinates
                print(f"Rounded x coordinate: {rounded_x}, Rounded y coordinate: {rounded_y}")
                
                # Extract and print relevant field number. 
                field_number = valid_file.split('(')[-1].split(')')[0]
                print(f"This galaxy was found in NIRCam Field {field_number}")

                galaxy_radius = None

                # Define the FITS files (different filters) to read based on the NIRCam Field where target galaxy lies in
                filenames = {
                    'f444w': f"f444w({field_number}).fits",
                    'f356w': f"f356w({field_number}).fits",
                    'f277w': f"f277w({field_number}).fits",
                    'f200w': f"f200w({field_number}).fits",
                    'f115w': f"f115w({field_number}).fits",
                    'f410m': f"f410m({field_number}).fits",
                    'f125w': f"f125w({field_number}).fits",
                    'f160w': f"f160w({field_number}).fits", 
                    'f606w': f"f606w({field_number}).fits",
                    'f814w': f"f814w({field_number}).fits"

                }

                # Determine galaxy radius in F444W filter
                if 'f444w' in filenames:

                    try:
                        # Extract cropped data and related uncertainty using above created function
                        cropped_data, cropped_uncertainty = self.get_fits_data(filenames['f444w'], rounded_x, rounded_y)

                        # Determine galaxy radius using above found cropped data
                        galaxy_radius = self.determine_galaxy_radius(cropped_data)
                        print(f"The galaxy radius using f444w filter is {galaxy_radius} pixels")

                    except Exception as e:
                        print(f"Error processing f444w data for galaxy ID {galaxy_id}: {e}")
                        continue

                try:
                    # Create an HDF5 file for storing data
                    with h5py.File(f"EGS_{galaxy_id:05d}.hdf5", "w") as f:

                        # Create a list of filters 
                        filter_groups = ['f444w', 'f356w', 'f277w', 'f200w', 'f115w', 'f410m', 'f125w', 'f160w', 'f606w', 'f814w']

                        # For loop to create groups in HDF5 file based on filters being analyzed
                        for group_name in filter_groups:
                            f.create_group(group_name)

                        # Initialize a flag to check if all cropped_data arrays are zero   
                        all_zero = True  

                        # Loop over each FITS file 
                        for group_name, filename in filenames.items():
                            try:
                                # Read and process FITS data. Supplying rounded_x and rounded_y here
                                cropped_data, cropped_uncertainty = self.get_fits_data(filename, rounded_x, rounded_y, galaxy_radius)

                                # If loop to check for empty image arrays
                                if np.all(cropped_data == 0):
                                    
                                    # Log onto external file if cropped_data is zero
                                    output_filename = 'Expansion_LogFile.csv'
                                    file_exists = os.path.isfile(output_filename)
                                    
                                    with open(output_filename, 'a', newline='') as csvfile:
                                        fieldnames = ['Galaxy_ID', 'Filter', 'Field Number', 'RA', 'Dec', 'Data Status']
                                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                        
                                        if not file_exists:
                                            writer.writeheader()
                                            
                                        status = "Data present" if not np.all(cropped_data == 0) else "Zero data"
                                        writer.writerow({
                                            'Galaxy_ID': galaxy_id, 
                                            'Filter': group_name, 
                                            'Field Number': field_number, 
                                            'RA': RA, 
                                            'Dec': Dec, 
                                            'Data Status': status
                                        })
                                    
                                    print(f"Zero data for galaxy ID {galaxy_id} in file {filename}. Logged and continuing.")
                                    
                                    continue
                                    
                                if np.all(cropped_data == 0, axis=0).any() or np.all(cropped_data == 0, axis=1).any():
                                    
                                    # Log even if cropped_data is zero
                                    output_filename = 'Expansion_LogFile.csv'
                                    file_exists = os.path.isfile(output_filename)
                                    
                                    with open(output_filename, 'a', newline='') as csvfile:
                                        fieldnames = ['Galaxy_ID', 'Filter', 'Field Number', 'RA', 'Dec', 'Data Status']
                                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                        
                                        if not file_exists:
                                            writer.writeheader()
                                            
                                        status = "Detector Edge" if not np.all(cropped_data == 0) else "Zero data"
                                        writer.writerow({
                                            'Galaxy_ID': galaxy_id, 
                                            'Filter': group_name, 
                                            'Field Number': field_number, 
                                            'RA': RA, 
                                            'Dec': Dec, 
                                            'Data Status': status
                                        })
                                    
                                    print(f"Galaxy ID {galaxy_id} in file {filename} is at the edge. Logged and continuing.")
                                    
                                    continue
                                    

                                # If any cropped_data is not zero, set the flag to False
                                all_zero = False  

                                # Create full image array
                                full_image = self.create_full_image(cropped_data, cropped_uncertainty)

                                # Store this as a dataset under the respective group name
                                f[group_name].create_dataset("image", data=full_image)

                            except Exception as e:
                                print(f"Error processing {group_name} data for galaxy ID {galaxy_id}: {e}")

                        if all_zero:
                            print(f"All cropped_data arrays are zero for Galaxy ID {galaxy_id}. Skipping this galaxy.")

                        else:
                            print(f"Data for Galaxy ID {galaxy_id} saved successfully.")

                except Exception as e:
                    print(f"Error creating HDF5 file for galaxy ID {galaxy_id}: {e}")
            else:
                print(f"No file contains the given RA and Dec for Galaxy ID {galaxy_id} within bounds.")

class LaguerreAmplitudes:
    def __init__(self, rscl, mmax, nmax, R=None, phi=None, mass=None, velocity=1.0):
        """
        Initialize the LaguerreAmplitudes object.

        Parameters:
        ----------
        rscl : float
            Scaling parameter for Laguerre functions.
        mmax : int
            Maximum order of the Laguerre expansion.
        nmax : int
            Maximum radial quantum number of the Laguerre expansion.
        R : array-like, optional
            Radial coordinates of pixels.
        phi : array-like, optional
            Angular coordinates of pixels.
        mass : array-like, optional
            Mass or intensity values of pixels.
        velocity : float, optional
            Velocity parameter used in Laguerre amplitude calculation.

        """
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = R
        self.phi = phi
        self.mass = mass
        self.velocity = velocity

        if self.R is not None and self.phi is not None:
            self.laguerre_amplitudes()

    def readsnapshot(self, datafile, groupname):
        """
        Read and preprocess image data from an HDF5 file.

        Parameters:
        -----------
        datafile : str
            Path to the HDF5 file containing image data.
        groupname : str
            Name of the group within the HDF5 file containing image data.

        Returns:
        --------
        rr : ndarray
            Radial coordinates of pixels. 
        pp : ndarray
            Angular coordinates of pixels.
        xpix : ndarray
            Meshgrid of x-pixel coordinates.
        ypix : ndarray
            Meshgrid of y-pixel coordinates.
        image_array : ndarray
            Processed image data.
        xdim : int
            Dimension of x-axis in the image array.
        ydim : int
            Dimension of y-axis in the image array.

        Notes:
        ------
        This method flattens the image data and applies a mask to exclude pixels based on specific metric.
        """
        with h5py.File(datafile, 'r') as f:
            
            # Assuming the initial FITS data is saved as 'image'
            image_data = f[groupname]['image'][:]
            
            # Extract image_array from image_data. This excludes the uncertainty, xarr and yarr arrays
            image_array = image_data[2]

            # Get the shape of image_array (2D shape)
            xdim, ydim = image_array.shape
            
            # Recreate xpixels and ypixels and set bounds
            xpixels = np.linspace(-xdim / 2, xdim / 2, xdim)
            ypixels = np.linspace(-ydim / 2, ydim / 2, ydim)
            
            # Create meshgrid and important to include the indexing here
            self.xpix, self.ypix = np.meshgrid(xpixels, ypixels, indexing='ij')

            # Set the radial and phi values
            rr, pp = np.sqrt(self.xpix**2 + self.ypix**2), np.arctan2(self.ypix, self.xpix)
            
            # Here mask excludes negative pixel contributions along with those outside radius = 100
            gvals = np.where((rr > 100.) | (image_array < 0))
            
            # Apply the mask by turning out of bound values to nan
            rr[gvals], pp[gvals], image_array[gvals] = np.nan, np.nan, np.nan
            self.R = rr.flatten()
            self.phi = pp.flatten()
            self.mass = image_array.flatten()

            # Returning xdim and ydim here also as they are used when plotting the contour maps
            return rr, pp, self.xpix, self.ypix, image_array, xdim, ydim

    def _gamma_n(self, nrange, rscl):
        """
        Compute the normalization constant 'gamma_n' for Laguerre functions.

        Parameters:
        -----------
        nrange : array-like
            Range of modal numbers for Laguerre functions.
        rscl : float
            Scale Length for Laguerre functions.

        Returns:
        --------
        gamma_n : ndarray
            Normalization constants for Laguerre functions.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        return (rscl / 2.) * np.sqrt(nrange + 1.)

    def _G_n(self, R, nrange, rscl):
        """
        Calculate the Laguerre basis functions G_n(R).

        Parameters:
        -----------
        R : array-like
            Radial coordinates.
        nrange : array-like
            Range of modal numbers for Laguerre functions.
        rscl : float
            Scale Length for Laguerre functions.

        Returns:
        --------
        G_n : ndarray
            Laguerre basis functions evaluated at radial coordinates R.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        laguerrevalues = np.array([eval_genlaguerre(n, 1, 2 * R / rscl) / self._gamma_n(n, rscl) for n in nrange])
        return np.exp(- R / rscl) * laguerrevalues

    def _n_m(self):
        """
        Compute the angular momentum normalization coefficients.

        Returns:
        --------
        nmvals : ndarray
            Normalization coefficients for angular momentum.

        Notes:
        ------
        This function is used internally within LaguerreAmplitudes class methods.
        """
        deltam0 = np.zeros(self.mmax)
        deltam0[0] = 1.0
        return np.power((deltam0 + 1) * np.pi / 2., -0.5)

    def laguerre_amplitudes(self):
        """
        Compute the Laguerre coefficients (coscoefs and sincoefs) for the given parameters.

        Notes:
        ------
        This method calculates the coefficients using the current values of R, phi, mass,
        and velocity attributes of the object.
        """
        G_j = self._G_n(self.R, np.arange(0, self.nmax), self.rscl)
        nmvals = self._n_m()
        cosm = np.array([nmvals[m] * np.cos(m * self.phi) for m in range(self.mmax)])
        sinm = np.array([nmvals[m] * np.sin(m * self.phi) for m in range(self.mmax)])

        self.coscoefs = np.nansum(cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity, axis=2)
        self.sincoefs = np.nansum(sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity, axis=2)

    def calculate_A1(self, n_min=0, n_max=None):
        """
        Calculate the A1 matrix, which is used for center finding.

        Parameters:
        -----------
        n_min : int, optional
            Minimum mode number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum mode number for Laguerre functions to include in calculation.

        Returns:
        --------
        A1 : float
            A1 matrix value, used for center finding.

        Notes:
        ------
        This method calculates A1 based on the specified range of Laguerre mode numbers.
        """
        if n_max is None:
            n_max = self.nmax
            
        # Begin by calculating cos and sine values for range of n values at m = 1. Returns  A1 expression
        c1n = self.coscoefs[1, n_min:n_max]
        s1n = self.sincoefs[1, n_min:n_max]

        return np.sqrt(np.sum(c1n**2 + s1n**2))

    def calculate_gradient(self, x, y, step=1e-3, n_min=0, n_max=None):
        """
        Calculate the gradient of A1 matrix at given coordinates (x, y) using finite differences.

        Parameters:
        -----------
        x : float
            x-coordinate for which gradient is calculated.
        y : float
            y-coordinate for which gradient is calculated.
        step : float, optional
            Step size for finite difference calculation.
        n_min : int, optional
            Minimum modal number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum modal number for Laguerre functions to include in calculation.

        Returns:
        --------
        gradient : ndarray
            Gradient of A1 matrix at coordinates (x, y).

        Notes:
        ------
        This method uses finite differences to approximate the gradient of A1 matrix
        at the specified coordinates (x, y).
        """
        original_R, original_phi = self.R.copy(), self.phi.copy()
        
        
        def compute_A1_at_shift(shift_x, shift_y):
            
            self.R = np.sqrt((self.xpix - shift_x)**2 + (self.ypix - shift_y)**2).flatten()
            self.phi = np.arctan2(self.ypix - shift_y, self.xpix - shift_x).flatten()
            self.laguerre_amplitudes()
            return self.calculate_A1(n_min, n_max)

        A1_x_plus_step = compute_A1_at_shift(x + step, y)
        A1_x_minus_step = compute_A1_at_shift(x - step, y)
        A1_y_plus_step = compute_A1_at_shift(x, y + step)
        A1_y_minus_step = compute_A1_at_shift(x, y - step)

        # Reset to original values
        self.R, self.phi = original_R, original_phi
        self.laguerre_amplitudes()

        # Compute the gradient. Note: Denominator here is 2 * step size
        dA1_dx = (A1_x_plus_step - A1_x_minus_step) / (2 * step)
        dA1_dy = (A1_y_plus_step - A1_y_minus_step) / (2 * step)

        return np.array([dA1_dx, dA1_dy])

    def find_center(self, initial_guess=(0, 0), tol=1e-2, max_iter=1000, n_min=0, n_max=None):
        
        """
        Find the center of the Laguerre expansion using Newton-Raphson method.

        Parameters:
        -----------
        initial_guess : tuple, optional
            Initial guess for center coordinates (x, y).
        tol : float, optional
            Tolerance for convergence of Newton-Raphson method.
        max_iter : int, optional
            Maximum number of iterations for Newton-Raphson method.
        n_min : int, optional
            Minimum modal number for Laguerre functions to include in calculation.
        n_max : int, optional
            Maximum modal number for Laguerre functions to include in calculation.

        Returns:
        --------
        x : float
            x-coordinate of the estimated center.
        y : float
            y-coordinate of the estimated center.
        reconstruction : ndarray
            Reconstructed image using Laguerre coefficients after center finding.

        Notes:
        ------
        This method iteratively applies Newton-Raphson method to refine the center
        estimate until tolerance limit is met.
        """
        
        # Initial guess for centre seperated into its coordinates
        x, y = initial_guess

        # Create a loop limited by a certain max number of iterations
        for i in range(max_iter):
            
            # Calculate the gradient for this guess for a standardised nmin and nmax
            gradient = self.calculate_gradient(x, y, n_min=n_min, n_max=n_max)
            step_direction = - gradient
            step_size = 1e-2
            
            # Update the initial guess
            x += step_size * step_direction[0]
            y += step_size * step_direction[1]

            # Break the loop only when its norm of negative gradient is < tolerance set
            if np.linalg.norm(step_direction) < tol:
                break

        # After finding the center, recalculate and set R and phi
        self.R = np.sqrt((self.xpix - x)**2 + (self.ypix - y)**2).flatten()
        self.phi = np.arctan2(self.ypix - y, self.xpix - x).flatten()
        
        # Recalculate the Laguerre amplitudes
        self.laguerre_amplitudes()

        # Perform the updated reconstruction
        reconstruction = self.laguerre_reconstruction(self.R, self.phi)

        return x, y, reconstruction

    def find_center_moments(self):
        """
        Calculate the center of mass (centroid) of the image's surface mass distribution.

        Returns:
        --------
        x_mass : float
            x-coordinate of the centroid.
        y_mass : float
            y-coordinate of the centroid.

        Notes:
        ------
        This method calculates the center of mass based on the mass distribution
        represented by the image mass array.
        """

        x_mass = np.nansum(self.xpix.flatten() * self.mass) / np.nansum(self.mass)
        y_mass = np.nansum(self.ypix.flatten() * self.mass) / np.nansum(self.mass)
        
        return x_mass, y_mass

    def update_orders(self, new_mmax, new_nmax):
        
        """
        Update the Laguerre expansion orders (mmax and nmax) and recompute Laguerre coefficients.

        Parameters:
        -----------
        new_mmax : int
            New maximum order of the Laguerre expansion.
        new_nmax : int
            New maximum radial quantum number of the Laguerre expansion.

        Notes:
        ------
        This method updates the Laguerre expansion orders, recalculates Laguerre coefficients,
        and applies a mask to exclude certain pixels based on specific criteria.
        """
        
        self.mmax, self.nmax = new_mmax, new_nmax
        self.laguerre_amplitudes()

        # Apply mask as before
        gvals = np.where((self.R > 100.) | (self.mass < 0))
        self.R[gvals], self.phi[gvals], self.mass[gvals] = np.nan, np.nan, np.nan

        # Visualise the effect of the mask applied
        plt.figure(figsize=(8, 6))
        plt.scatter(self.xpix, self.ypix, c=self.mass, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Mask values')
        plt.title('Visualization of Mask')
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.show()

    def laguerre_reconstruction(self, rr, pp):
        """
        Reconstruct the original image using Laguerre coefficients.

        Parameters:
        -----------
        rr : array-like
            Radial coordinates for reconstruction.
        pp : array-like
            Angular coordinates for reconstruction.

        Returns:
        --------
        reconstruction : ndarray
            Reconstructed image using Laguerre coefficients.

        Notes:
        ------
        This method reconstructs the original image using the Laguerre coefficients
        calculated from the Laguerre expansion.
        """
        nmvals = self._n_m()
        G_j = self._G_n(rr, np.arange(0, self.nmax), self.rscl)

        fftotal = sum(
            self.coscoefs[m, n] * nmvals[m] * np.cos(m * pp) * G_j[n]
            + self.sincoefs[m, n] * nmvals[m] * np.sin(m * pp) * G_j[n]
            for m in range(self.mmax) for n in range(self.nmax)
        )

        return 0.5 * fftotal

    def compute_ratio(self, rscl, mmax, nmax, rval, phi, snapshotflat):
        """
        Compute a ratio metric using Laguerre amplitudes.

        Parameters:
        -----------
        rscl : float
            Scaling parameter for Laguerre functions.
        mmax : int
            Maximum order of the Laguerre expansion.
        nmax : int
            Maximum radial quantum number of the Laguerre expansion.
        rval : float
            Radial coordinate for the Laguerre expansion.
        phi : float
            Angular coordinate for the Laguerre expansion.
        snapshotflat : array-like
            Flattened snapshot image data.

        Returns:
        --------
        ratio : float
            Ratio metric computed using Laguerre amplitudes.
        """
        laguerre = LaguerreAmplitudes(rscl, mmax, nmax, rval, phi, snapshotflat)
        abs_coscoefs_0 = abs(laguerre.coscoefs[0, 0])
        norm_coscoefs_1 = np.linalg.norm(laguerre.coscoefs[0, 1:])
        return norm_coscoefs_1 / abs_coscoefs_0
    
# Constants
f444w_group = 'f444w'
rscl_initial = 10
mmax_initial = 2
nmax_initial = 10
rscl_values = np.linspace(1, 20, 100)
new_mmax = 8
new_nmax = 24

def process_galaxy(galaxy_id, fits_files, filename):
    """
    Process a galaxy by extracting data, computing Laguerre amplitudes, and generating plots.

    Parameters:
    galaxy_id (int): The ID of the galaxy to be processed.
    fits_files (list of str): List of paths to the FITS files.
    filename (str): Path to the mass catalog CSV file.

    Returns:
    - Centre and scale length details
    - Mask Image
    - Coefficient Chart
    - Coefficients, centre and scale length saved on HDF5 file
    - Display 4 contour maps including uncertainty maps
    """
    
    extractor = GalaxyDataExtractor(fits_files, filename, [galaxy_id])
    extractor.process_galaxies()

    # Open the HDF5 file in appending mode
    with h5py.File(f"EGS_{galaxy_id:05d}.hdf5", "a") as f:
        
        if f444w_group not in f:
            f.create_group(f444w_group)

        f444w = f[f444w_group]

        # Create an instance of LaguerreAmplitudes and read snapshot data
        L = LaguerreAmplitudes(rscl_initial, mmax_initial, nmax_initial)
        
        # Try and except block done here to bypass galaxy ID's which have empty image arrays. 
        try:
            rr, pp, xpix, ypix, fixed_image, xdim, ydim = L.readsnapshot(f"EGS_{galaxy_id:05d}.hdf5", f444w_group)
        
        except KeyError as e:
            print(f"Error processing galaxy ID {galaxy_id}: {e}")
            # Skip to the next galaxy ID
            return  

        # Calculate the Laguerre amplitudes
        L.laguerre_amplitudes()

        # Find center using moments and optimization method
        x_mass, y_mass = L.find_center_moments()
        center_x, center_y, _ = L.find_center(initial_guess=(x_mass, y_mass))

        # Test a range of rscl values to find the best one
        min_ratio = float('inf')
        best_rscl = None
        for rscl in rscl_values:
            ratio = L.compute_ratio(rscl, mmax_initial, nmax_initial, rr, pp, fixed_image)
            min_ratio_index = np.argmin(np.abs(ratio))
            if np.abs(ratio[min_ratio_index]) < np.abs(min_ratio):
                min_ratio = ratio[min_ratio_index]
                best_rscl = rscl

        # Set the best rscl value
        L.rscl = best_rscl

        print(f"Center of mass at x = {center_x}, y = {center_y}")
        print(f"New rscl value: {best_rscl}")

        # Update orders and recalculate the Laguerre amplitudes
        L.update_orders(new_mmax, new_nmax)
        rr, pp = np.sqrt((xpix - center_x)**2 + (ypix - center_y)**2), np.arctan2(ypix - center_y, xpix - center_x)
        L.laguerre_amplitudes()

        # Perform the reconstruction
        reconstruction = L.laguerre_reconstruction(rr.flatten(), pp.flatten())

        # Save the expansion coefficients to the HDF5 file
        dset = f444w.create_dataset(f"{galaxy_id}/expansion", data=np.stack([L.coscoefs, L.sincoefs]))

        # Save the center and scale length values
        f444w[f"{galaxy_id}/expansion"].attrs['centre'] = [center_x, center_y]
        f444w[f"{galaxy_id}/expansion"].attrs['scale_length'] = best_rscl

        # Calculate the amplitude of the coefficients and create the plot using imshow
        amplitudes = np.sqrt(L.coscoefs**2 + L.sincoefs**2).T
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        c = ax.imshow(amplitudes, cmap='Blues', norm=LogNorm(), aspect='auto')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel('Harmonics (m)')
        ax.set_ylabel('Radial Nodes (n)')
        ax.set_title('Amplitude of Fourier-Laguerre Coefficients')
        ax.set_xticks(np.arange(L.mmax))
        ax.set_yticks(np.arange(L.nmax))
        plt.tight_layout()
        plt.savefig(f'coeff(f444w)_{galaxy_id:05d}.png', dpi=600)
        plt.show()

        # Create 4 subplots for visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
        cval = np.linspace(-5., 1., 32)

        # Plot in the following order: Original Image, Expanded Image, Relative Uncertainty and Absolute Uncertainty
        ax1.contourf(xpix, ypix, np.log10(fixed_image), cval, cmap=plt.cm.Greys)
        ax2.contourf(xpix - center_x, ypix - center_y, np.log10(reconstruction.reshape(xdim, ydim)), cval, cmap=plt.cm.Greys)
        ax3.contourf(xpix - center_x, ypix - center_y, (reconstruction.reshape(xdim, ydim) - fixed_image) / fixed_image, np.linspace(-.25, .25, 100), cmap=plt.cm.Greys)
        ax4.contourf(xpix - center_x, ypix - center_y, abs(reconstruction.reshape(xdim, ydim) - fixed_image), np.linspace(-.25, .25, 100), cmap=plt.cm.Greys)

        for ax, title in zip([ax1, ax2, ax3, ax4], ['log surface density', 'expanded surface density', 'relative uncertainty', 'absolute uncertainty']):
            ax.set_title(title)
            ax.axis([-xdim/2, xdim/2, -xdim/2, xdim/2])
            ax.set_xticklabels(())
            ax.set_yticklabels(())
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(axis="y", which='both', direction="in")
            ax.tick_params(axis="x", which='both', direction="in", pad=5)
            ax.scatter(0.0 if ax != ax1 else center_x, 0.0 if ax != ax1 else center_y, color='red', marker='x', label='(0,0)' if ax != ax1 else 'True Center')

        plt.tight_layout()
        plt.savefig(f'expand(f444w)_{galaxy_id:05d}.png', dpi=600)
        plt.show()

def analyze_filter(galaxyid, filter_name, mmax_initial, nmax_initial, rscl_initial, new_mmax, new_nmax, group_name='f444w'):
    """
    Perform the Laguerre expansion analysis for a given galaxy and filter.

    Parameters:
    galaxyid (int): The ID of the galaxy to be processed.
    filter_name (str): The name of the filter (e.g., 'f356w', 'f277w').
    mmax_initial (int): The initial maximum order of the harmonic expansion.
    nmax_initial (int): The initial maximum order of the radial expansion.
    rscl_initial (float): The initial scale length parameter.
    new_mmax (int): The updated maximum order of the harmonic expansion.
    new_nmax (int): The updated maximum order of the radial expansion.
    group_name (str, optional): The name of the group used to find the initial center and scale length. Default is 'f444w'.

    Returns:
    - Centre and scale length details
    - Mask Image
    - Coefficient Chart
    - Coefficients, centre and scale length saved on HDF5 file
    - Display 4 contour maps
    """
    
    try: 
        print(f"Expanding galaxy {galaxyid} in filter {filter_name}")

        # Open file in append mode to add the expansion data
        with h5py.File(f"EGS_{galaxyid:05d}.hdf5", "a") as f:
            filter_group = f[filter_name]

            # Create an instance of LaguerreAmplitudes
            L = LaguerreAmplitudes(rscl_initial, mmax_initial, nmax_initial)

            # Read the snapshot data for the current filter
            try:
                rr, pp, xpix, ypix, fixed_image, xdim, ydim = L.readsnapshot(f"EGS_{galaxyid:05d}.hdf5", filter_name)
            
            except KeyError as e:
                print(f"Error processing filter {filter_name} for galaxy ID {galaxyid}: {e}")
                return  # Skip to the next filter

            # Calculate the Laguerre amplitudes
            L.laguerre_amplitudes()

            # Use the center of the galaxy from the specified group as the initial guess
            center_x, center_y = f[group_name][f"{galaxyid}/expansion"].attrs['centre']

            # Find the center using the center of the galaxy from the specified group as the initial guess
            center_x, center_y, reconstruction = L.find_center(initial_guess=(center_x, center_y), tol=1e-5, max_iter=1000)

            # Choose the best rscl value obtained from Method 2 using the specified group
            new_rscl = f[group_name][f"{galaxyid}/expansion"].attrs['scale_length']
            L.rscl = new_rscl

            print(f"Center of mass at x = {center_x}, y = {center_y}")
            print(f"New rscl value: {new_rscl}")

            # Update orders and recalculate the Laguerre amplitudes
            L.update_orders(new_mmax, new_nmax)
            rr, pp = np.sqrt((xpix - center_x)**2 + (ypix - center_y)**2), np.arctan2(ypix - center_y, xpix - center_x)
            L.laguerre_amplitudes()

            # Perform the reconstruction
            reconstruction = L.laguerre_reconstruction(rr.flatten(), pp.flatten())

            # Save the expansion coefficients to the HDF5 file 
            dset = filter_group.create_dataset(f"{galaxyid}/expansion", data=np.stack([L.coscoefs, L.sincoefs]))

            # Save the center and scale length values
            filter_group[f"{galaxyid}/expansion"].attrs['centre'] = [center_x, center_y]
            filter_group[f"{galaxyid}/expansion"].attrs['scale_length'] = new_rscl

            # Calculate the amplitude of the coefficients
            amplitudes = np.sqrt(L.coscoefs**2 + L.sincoefs**2).T

            # Create the plot for coefficients
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            c = ax.imshow(amplitudes, cmap='Blues', norm=LogNorm(), aspect='auto')
            fig.colorbar(c, ax=ax)
            ax.set_xlabel('Harmonics (m)')
            ax.set_ylabel('Radial Nodes (n)')
            ax.set_title('Amplitude of Fourier-Laguerre Coefficients')
            ax.set_xticks(np.arange(L.mmax))
            ax.set_yticks(np.arange(L.nmax))
            plt.tight_layout()
            plt.savefig(f'coeff({filter_name})_{galaxyid:05d}.png', dpi=600)
            plt.show()
            
            # Create 4 subplots for visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
            cval = np.linspace(-5., 1., 32)

            # Plot in the following order: Original Image, Expanded Image, Relative Uncertainty and Absolute Uncertainty
            ax1.contourf(xpix, ypix, np.log10(fixed_image), cval, cmap=plt.cm.Greys)
            ax2.contourf(xpix - center_x, ypix - center_y, np.log10(reconstruction.reshape(xdim, ydim)), cval, cmap=plt.cm.Greys)
            ax3.contourf(xpix - center_x, ypix - center_y, (reconstruction.reshape(xdim, ydim) - fixed_image) / fixed_image, np.linspace(-.25, .25, 100), cmap=plt.cm.Greys)
            ax4.contourf(xpix - center_x, ypix - center_y, abs(reconstruction.reshape(xdim, ydim) - fixed_image), np.linspace(-.25, .25, 100), cmap=plt.cm.Greys)

            for ax, title in zip([ax1, ax2, ax3, ax4], ['log surface density', 'expanded surface density', 'relative uncertainty', 'absolute uncertainty']):
                ax.set_title(title)
                ax.axis([-xdim/2, xdim/2, -xdim/2, xdim/2])
                ax.set_xticklabels(())
                ax.set_yticklabels(())
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis="y", which='both', direction="in")
                ax.tick_params(axis="x", which='both', direction="in", pad=5)
                ax.scatter(0.0 if ax != ax1 else center_x, 0.0 if ax != ax1 else center_y, color='red', marker='x', label='(0,0)' if ax != ax1 else 'True Center')

            plt.tight_layout()
            plt.savefig(f'expand({filter_name})_{galaxyid:05d}.png', dpi=600)
            plt.show()
            
            print('\n' + '=' * 20 + ' NEXT FILTER ' + '=' * 20 + '\n')

      
    except Exception as e:
        print(f"Error processing galaxy ID {galaxyid} in filter {filter_name}: {e}")
        return  # Skip to the next galaxy or filter
    
# Constants to use for the function
filters = ['f356w', 'f277w', 'f200w', 'f115w', 'f410m', 'f125w', 'f160w', 'f606w', 'f814w']  
mmax_initial = 2
nmax_initial = 10
rscl_initial = 10
new_mmax = 8
new_nmax = 24
