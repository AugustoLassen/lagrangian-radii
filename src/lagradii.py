"""
Author: A. Lassen
Last modification: August 1 2025

------------------- Change log
14/07/25: Fixed incorrect calculation of kpc/pixel parameters
15/07/25: SSD calculation has been correctly implemented into class
24/07/25: Updated methods for inclination correction
29/07/25: Modularization of calc_lagradii() method
          Added fill_zeros() utility function for error handling for edge cases in angular bin processing
01/08/25: Updated method to calculate SSD, now including handling of uncertainties
"""
__version__ = "0.3.1"

import numpy as np
import pandas as pd
import astropy.units as u

from tqdm import tqdm
from uncertainties import unumpy

# ---- Cosmology parameters assumed
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# ----

def fill_zeros(indices):
    indices = indices.copy()  # avoid modifying the original array
    for i in range(1, len(indices)):
        if indices[i] == 0:
            indices[i] = indices[i - 1]
    return indices

class AngSlicing():
    __version__ = __version__

    def __init__(self, imdata, center, mask=None, **kwargs):
        ### Handle keyword arguments and default options
        self.__set_params__(**kwargs)

        ### Unpack central coordinates. Add image as an attribute
        self.xc, self.yc = center
        self.imdata = imdata.copy()

        ### Validate bin width
        self.nbin = 360. / self.bin_width
        assert self.nbin % 1 == 0.0, "The angle bin widht must be a multiple of 360."
        self.nbin = np.int32(self.nbin)

        ### After validation, add convenience attribute
        self.angles = np.arange(0., 360., self.bin_width, dtype=np.float32)

        ### Apply mask
        self._apply_mask(mask)

    def __set_params__(self, **kwargs):
        defaults = {
            # Bin width (degrees) of the slices.
            # Obs.: Must be an integer multiple of 360.
            "bin_width": 20.,

            # Pixels values in the image below this threshold are masked
            "min_value": 0.0}

        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)

    def _apply_mask(self, mask):
        if mask is None: self.mask = np.zeros(self.imdata.shape, dtype=bool)
        else: self.mask = mask

        self.imdata[self.imdata < self.min_value] = np.nan
        self.imdata[self.mask] = np.nan

        return None
    
    def slicing(self):
        ### Use unmasked pixels to calculate angles, shifting the origin of ref. frame to center.
        yarray, xarray = np.where(~self.mask)
        dy = yarray - self.yc
        dx = xarray - self.xc

        angles = np.rad2deg(np.arctan2(dx, dy))

        # Shifting angle domain from [-pi, pi] to [0, 2pi]
        angles += 180. 
        
        ### Initialize table
        t = {"x": xarray, "y": yarray, "dx": dx, "dy": dy,
             "theta": angles, "values": []}
        
        for iy, ix in zip(yarray, xarray):
            t["values"].append(self.imdata[iy, ix])

        tab = pd.DataFrame(t)
        abins = np.linspace(0, 360, self.nbin + 1)
        tab["bin"] = np.digitize(tab["theta"], abins, right=False) - 1

        return tab

class Lag_radii():
    __version__ = __version__
    
    def __init__(self, im_dict, center, mask_dict, z=None, quiet=False, **kwargs):
        ### Handle keyword arguments and default options
        self.__set_params__(**kwargs)
        self.quiet = quiet

        ### Unpack central coordinates
        self.xc, self.yc = center

        ### If redshift is provided, calculate physical scale (kpc/arcsec) at this given distance
        self.z = z
        if self.z is not None:
            ps_kpc = cosmo.kpc_proper_per_arcmin(self.z).value/60.
            self.kpc_per_pixel = self.px_scale * ps_kpc # [kpc/pixel]
        else:
            self.kpc_per_pixel = 0.0

        ### Validate required keys in both dictionaries
        required_keys = ["young", "int"]
        for dict_name, d in [("im_dict", im_dict), ("mask_dict", mask_dict)]:
            missing = np.array(required_keys)[~np.isin(required_keys, list(d.keys()))]
            if len(missing) > 0:
                raise ValueError(f"{dict_name} missing required keys: {missing.tolist()}")

        ### Add images and masks as attributes of the class
        self.imy, self.imi = im_dict["young"], im_dict["int"]
        self.m1, self.m2 = mask_dict["young"], mask_dict["int"]

        assert self.imy.ndim == self.imi.ndim == self.m1.ndim == self.m2.ndim == 2,\
            "Input arrays are not images! (i.e. dim != 2)"
        assert self.imy.shape == self.imi.shape == self.m1.shape == self.m2.shape,\
            "Shape of input arrays do not match!"

        # Ensure mask is applied to input images
        self.imy[self.m1], self.imi[self.m2] = np.nan, np.nan

        ### Get initial tables
        self.taby, self.tabi = self._make_tab()

    def __set_params__(self, **kwargs):
        defaults = {
            # Bin width (degrees) of the cones. Obs.: Must be an integer multiple of 360.
            "bin_width": 20.,

            # Pixels values in the image below this threshold are masked
            "min_value": 0.0,

            # Flux percentiles used to define radial sectors in the image
            "q": np.array([50., 75., 90., 95., 99.]),

            # Lagrangian radius limitation (px)
            "rlim": None,

            # CCD pixels scale (arcsec/pixel). Relevant only if physical units are meant to be used.
            "px_scale": 0.2}
        
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)

    def _make_tab(self):
        ### Start applying angular slicing in each image
        aslice = {}
        aslice["young"] = AngSlicing(self.imy, (self.xc, self.yc), mask=self.m1,
                                     bin_width=self.bin_width, min_value=self.min_value)
        aslice["int"] = AngSlicing(self.imi, (self.xc, self.yc), mask=self.m2,
                                   bin_width=self.bin_width, min_value=self.min_value)
        
        ### Inherit attributes
        self.nbin, self.angles = aslice["young"].nbin, aslice["young"].angles

        ### Get initial tables
        taby, tabi = aslice["young"].slicing(), aslice["int"].slicing()

        ### Add a column with the distance from the center.
        dr1 = np.sqrt(taby["dx"]**2. + taby["dy"]**2.)
        dr2 = np.sqrt(tabi["dx"]**2. + tabi["dy"]**2.)
        taby.insert(loc=taby.columns.get_loc("dy")+1, column="dr", value=dr1)
        tabi.insert(loc=tabi.columns.get_loc("dy")+1, column="dr", value=dr2)

        ### If a redshift is provided, add a column with the corresponding physical distance (kpc)
        if self.z is not None:
            taby.insert(loc=taby.columns.get_loc("dr")+1, column="dr_kpc", value = self.kpc_per_pixel * dr1)
            tabi.insert(loc=tabi.columns.get_loc("dr")+1, column="dr_kpc", value = self.kpc_per_pixel * dr2)

        return taby,tabi

    def calc_lagradii(self, tab):
        ### Start by sorting table by angle value        
        tab = tab.sort_values(by="theta", ascending=True)
        tab.reset_index(drop=True, inplace=True)

        ### Create the outputs
        shape = ((self.nbin, self.q.size))
        xslice = np.zeros(shape, dtype=np.float32)
        yslice, rslice, rslice_phys = np.zeros_like(xslice), np.zeros_like(xslice), np.zeros_like(xslice)

        ### Iterate over angle bins
        angle_bins = np.arange(0, self.nbin)
        if self.quiet:
            for i in angle_bins:
                xslice[i, :], yslice[i, :], rslice[i, :], rslice_phys[i, :] = self._process_abin(i, tab)

        else:
            for i in tqdm(angle_bins, desc="Iterating over angle bins", total=self.nbin):
                xslice[i, :], yslice[i, :], rslice[i, :], rslice_phys[i, :] = self._process_abin(i, tab)        
        
        return {"x": xslice,
                "y": yslice,
                "r": rslice,
                "rphys": rslice_phys}
    
    def _process_abin(self, i, tab):
        ### Start by initalizing the outputs
        xslice, yslice, rslice, rslice_phys = [], [], [], []

        ### Get sub-table for a given angular bin
        tab_i = tab.loc[tab["bin"] == i].copy()
        tab_i.dropna(inplace=True)

        ### Sort sub-table by the distance (r) from the center
        tab_i = tab_i.sort_values("dr")
        tab_i.reset_index(drop=True, inplace=True)

        ### Calculate the total value within this particular angular slice
        if self.rlim is None: tot_flux = tab_i["values"].sum()
        else: tot_flux = tab_i["values"].loc[tab_i["dr"] <= self.rlim].sum()

        ### Iterate over the rows of sorted sub-table
        ### Obs.: Quartiles expressed in terms of flux percentages
        qflux = self.q * (tot_flux/100.)

        f,k = 0,0
        indices = np.zeros(self.q.size, dtype=np.int32)
        for j, row in tab_i.iterrows():
            f += row["values"]
            if f >= qflux[k]:
                indices[k] = j
                k += 1
            
            if k > self.q.size - 1: break
        if tab_i.shape[0] <= indices.size: indices = fill_zeros(indices)

        ### Get list of tables, sliced at different Lag. radii                
        theta1 = i * self.bin_width
        theta2 = (i + 1) * self.bin_width
        mean_theta_rad = 0.5*(theta1 + theta2)

        if tab_i.empty:
            xslice.append(0.)
            yslice.append(0.)
            rslice.append(0.)
            rslice_phys.append(0.)

        else:
            rslic_tables = [tab_i.iloc[np.arange(0, r+1)] for r in indices]
            for rslic_tab in rslic_tables:
                rslic = rslic_tab["dr"].max()

                rslice.append(rslic)
                xslice.append(rslic * (-1.) * np.sin(np.deg2rad(mean_theta_rad)))
                yslice.append(rslic * (-1.) * np.cos(np.deg2rad(mean_theta_rad)))

                if np.isin("dr_kpc", list(tab.columns)): rslice_phys.append(rslic_tab["dr_kpc"].max())
                else: rslice_phys.append(0.)

        return xslice, yslice, rslice, rslice_phys

    def rot_and_stretch(self, x, y, PA, inc, ang_type="deg"):
        if ang_type == "deg":
            PA = np.radians(PA)
            inc = np.radians(inc)

        ### Rotation
        xnew = x * np.cos(PA) - y * np.sin(PA)
        ynew = x * np.sin(PA) + y * np.cos(PA)

        ### Stretching
        yf = ynew.copy()
        xf = xnew / np.cos(inc)
        rf = np.sqrt(xf**2. + yf**2.)

        return xf, yf, rf

    def deprojection(self, yslice, islice, PA, inc, ang_type="deg"):
        ### Validate input dictionaries. Convert angles to radians if they were provided in degrees
        assert list(yslice.keys()) == list(islice.keys()), "yslice and islice keys do not match"

        if ang_type == "deg":
            PA = np.radians(PA)
            inc = np.radians(inc)

        ### Initialize outputs
        yslice_icorr, islice_icorr = {}, {}
        for key in yslice.keys():
            yslice_icorr[key] = np.zeros_like(yslice[key])
            islice_icorr[key] = np.zeros_like(islice[key])

        ### Apply deprojection by rotating and stretching x,y coordinates of the cones
        for i in range(self.q.size):
            deproj_x, deproj_y, deproj_r = self.rot_and_stretch(yslice["x"][:, i], yslice["y"][:, i],
                                                                PA, inc, ang_type="radians")
            yslice_icorr["x"][:, i] = deproj_x
            yslice_icorr["y"][:, i] = deproj_y
            yslice_icorr["r"][:, i] = deproj_r

        ### Also convert deprojected radii values to physical units
        yslice_icorr["rphys"] = self.kpc_per_pixel *  np.sqrt(yslice_icorr["x"]**2. + yslice_icorr["y"]**2.)
        islice_icorr["rphys"] = self.kpc_per_pixel *  np.sqrt(islice_icorr["x"]**2. + islice_icorr["y"]**2.)

        return yslice_icorr, islice_icorr

    # def deprojection(self, PA, inc, ang_type="deg"):
    #     if ang_type == "deg":
    #         PA = np.radians(PA)
    #         inc = np.radians(inc)

    #     xcone_int, ycone_int, rcone_int, rphys_int = self.int_cones
    #     xcone_young, ycone_young, rcone_young, rphys_young = self.young_cones

    #     ### Initialize outputs
    #     xcy_icorr, xci_icorr = np.zeros_like(xcone_young), np.zeros_like(xcone_int)
    #     ycy_icorr, yci_icorr = np.zeros_like(ycone_young), np.zeros_like(ycone_int)
    #     rcy_icorr, rci_icorr = np.zeros_like(rcone_young), np.zeros_like(rcone_int)

    #     ### Apply deprojection by rotating and stretching x,y coordinates of the cones
    #     for i in range(self.q.size):
    #         xcy_icorr[:, i], ycy_icorr[:, i], rcy_icorr[:, i] = \
    #             self.rot_and_stretch(xcone_young[:,i], ycone_young[:,i],PA, inc, ang_type="radians")
    #         xci_icorr[:, i], yci_icorr[:, i], rci_icorr[:, i] = \
    #             self.rot_and_stretch(xcone_int[:,i], ycone_int[:,i], PA, inc, ang_type="radians")

    #     ### Also convert deprojected rcones to physical units
    #     rcy_phys_icorr = np.sqrt((self.kpc_per_pixel * xcy_icorr)**2. + ((self.kpc_per_pixel * ycy_icorr)**2.))
    #     rci_phys_icorr = np.sqrt((self.kpc_per_pixel * xci_icorr)**2. + ((self.kpc_per_pixel * yci_icorr)**2.))

    #     return [(xcy_icorr, ycy_icorr, rcy_icorr, rcy_phys_icorr),
    #             (xci_icorr, yci_icorr, rci_icorr, rci_phys_icorr)]

    def calculate_SSD(self, yslice, islice, re_arcsec, start=1, errors=False):
        ### Attribute unit to Re
        re_arcsec *= u.arcsec

        ### For normalization, we'll convert Lagrangian radii from kpc to arcsec
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(self.z).to("kpc/arcsec")

        ### Calculate the SSD, iterating over Lagrangian radii
        ssd = []
        if errors: ssd_err = []
        for k in range(start, yslice["rphys"].shape[1]):
            ### Radii at a fixed flux percentile
            if not errors:
                y, i = yslice["rphys"][:, k], islice["rphys"][:, k]

                diff = y - i
                ssd.append(np.sum(np.abs(diff)))
            else:
                y = unumpy.uarray(yslice["rphys"][:, k], yslice["rphys_err"][:, k])
                i = unumpy.uarray(islice["rphys"][:, k], islice["rphys_err"][:, k])

                diff = y - i
                ssd.append(np.sum(np.abs(diff)).n)
                ssd_err.append(np.sum(np.abs(diff)).std_dev)
                
        ### Normalize the SSD by the effective radius Re
        if not errors:
            # SSD(kpc) --> arcsec and then divide them by Re (arcsec)
            SSD_norm = ((np.sum(ssd) * u.kpc) / kpc_per_arcsec) / re_arcsec
            assert SSD_norm.unit == u.dimensionless_unscaled, f"After normalization, SSD should be dimensionless!"
            
            SSD_norm = SSD_norm.value
            return SSD_norm

        else:
            SSD_norm = (np.sum(unumpy.uarray(ssd, ssd_err)) / kpc_per_arcsec.value) / re_arcsec.value
            return SSD_norm.n, SSD_norm.std_dev

