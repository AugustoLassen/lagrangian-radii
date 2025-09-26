"""
Author: A. Lassen
Last modification: September 26th, 2025

------------------- Change log
14/07/25: Fixed incorrect calculation of kpc/pixel parameters
15/07/25: SSD calculation has been correctly implemented into class
24/07/25: Updated methods for inclination correction
29/07/25: Modularization of calc_lagradii() method
          Added fill_zeros() utility function for error handling of edge cases in angular bin processing
01/08/25: Updated method to calculate SSD, now including handling of uncertainties
26/09/25: Code version 0.3.1 --> 0.4. Summary of the alterations:
          - Added support for error arrays (err_dict)
          - Implemented bootstrap_cumsum() for uncertainty estimation of Lagrangian radii
          - Refactored _process_abin() to use cumulative interpolation to derive Lagrangian radii
          - calc_lagradii() now returns uncertainties (r_err, rphys_err) along with radii
          - Deprecated fill_zeros() function
          - Improved validation process of inputs
"""
__version__ = "0.4"

import numpy as np
import pandas as pd
import astropy.units as u

from tqdm import tqdm
from uncertainties import unumpy

# ---- Cosmology parameters assumed
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# ----

# ===================================== //
def bootstrap_cumsum(tab, percentiles, xcol_name="dr", ycol_name="values",
                     nboot=2000, sig_y=None, random_state=None, mode="normal"):
    ### Convert percentiles to fractionals, and set up random generator
    ps = 0.01 * percentiles
    rng = np.random.default_rng(random_state)

    ### Unpack x,y arrays from input table columns
    assert np.all(np.isin([xcol_name, ycol_name], tab.columns)), \
           f"Columns {xcol_name} and/or {ycol_name} are not included in input table"
    x = tab[xcol_name].to_numpy()
    y = tab[ycol_name].to_numpy()

    ### If an y-error is not provided, assume it is sqrt(y)
    if sig_y is None:
        sig_y = np.sqrt(np.clip(y, 0, None))

    ### Bootstrapping on cumulative sum
    samples = {q: [] for q in [f"q{i+1}" for i in range(ps.size)]}
    for j in range(nboot):
        if mode == "poisson": pert = rng.poisson(np.clip(y, 0, None))
        else:
            pert = rng.normal(y, np.maximum(1e-12, sig_y))
            pert = np.clip(pert, 0., None) # ensure non-negative values

        tot = pert.sum()
        if tot == 0:
            for key in samples.keys(): samples[key].append(np.nan)
            continue

        cum = np.cumsum(pert) / tot
        for index, q in enumerate(ps):
            if q < cum[0] or q > cum[-1]: samples[f"q{index+1}"].append(np.nan)
            else:
                samples[f"q{index+1}"].append(np.interp(q, cum, x))
    
    ### Return chain with all disturbed arrays and results
    chains = {}
    for key in samples.keys():
        arr = np.asarray(samples[key])
        arr = arr[~np.isnan(arr)]

        if arr.size == 0: chains[key] = (np.nan, np.nan, arr)
        else:
            chains[key] = (np.mean(arr), np.std(arr, ddof=1), arr)

    return chains
# ===================================== //

class AngSlicing():
    __version__ = __version__

    def __init__(self, imdata, center, mask=None, err=None, **kwargs):
        ### Handle keyword arguments and default options
        self.__set_params__(**kwargs)

        ### Unpack central coordinates. Add image as an attribute
        self.xc, self.yc = center
        self.imdata = imdata.copy()

        self.err = err
        if self.err is not None:
            assert np.logical_and(self.err.ndim == self.imdata.ndim,
                                  self.err.shape == self.imdata.shape), \
                                  "Error array must be identical to input image array!"
        
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

        if self.err is not None:
            self.err[np.isnan(self.imdata)] = np.nan

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
        if self.err is not None: t["values_err"] = list()
        
        for iy, ix in zip(yarray, xarray):
            t["values"].append(self.imdata[iy, ix])
            if self.err is not None: t["values_err"].append(self.err[iy, ix])

        tab = pd.DataFrame(t)
        abins = np.linspace(0, 360, self.nbin + 1)
        tab["bin"] = np.digitize(tab["theta"], abins, right=False) - 1

        return tab

class Lag_radii():
    __version__ = __version__
    
    def __init__(self, im_dict, center, mask_dict, err_dict=None, z=None, quiet=False, **kwargs):
        ### Handle keyword arguments and default options
        self.__set_params__(**kwargs)
        self.quiet = quiet

        if err_dict is not None: self.has_err=True
        else: self.has_err=False

        ### Unpack central coordinates
        self.xc, self.yc = center
        
        ### If redshift is provided, calculate physical scale (kpc/arcsec) at this given distance
        self.z = z
        if self.z is not None:
            ps_kpc = cosmo.kpc_proper_per_arcmin(self.z).to("kpc/arcsec").value
            self.kpc_per_pixel = self.px_scale * ps_kpc # [kpc/pixel]
        else:
            self.kpc_per_pixel = 0.0

        ### Validate required keys in all dictionaries
        self.__validate_dicts__(im_dict, mask_dict, err_dict)

        ### Add images and masks as attributes of the class and validate them afterwards
        self.imy, self.imi = im_dict["young"], im_dict["int"]
        self.m1, self.m2 = mask_dict["young"], mask_dict["int"]

        self.__validate_arrays__(err_dict)

        # Ensure mask is applied to input images
        self.imy[self.m1], self.imi[self.m2] = np.nan, np.nan
        if self.has_err:
            self.erry[self.m1], self.erri[self.m2] = np.nan, np.nan

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

    def __validate_dicts__(self, im_dict, mask_dict, err_dict, required_keys=["young", "int"]):
        for dict_name, d in [("im_dict", im_dict),
                             ("mask_dict", mask_dict)]:
            
            missing = np.array(required_keys)[~np.isin(required_keys, list(d.keys()))]
            if len(missing) > 0:
                raise ValueError(f"{dict_name} missing required keys: {missing.tolist()}")

        if self.has_err:
            missing = np.array(required_keys)[~np.isin(required_keys, list(err_dict.keys()))]
            if len(missing) > 0:
                raise ValueError(f"err_dict missing required keys: {missing.tolist()}")
            
        return None
    
    def __validate_arrays__(self, err_dict):
        shapes = [self.imy.shape, self.imi.shape, self.m1.shape, self.m2.shape]
        dim_array = np.array([self.imy.ndim, self.imi.ndim, self.m1.ndim, self.m2.ndim])
        
        if self.has_err:
            self.erry, self.erri = err_dict["young"], err_dict["int"]
            shapes.extend([self.erry.shape, self.erri.shape])
            dim_array = np.append(dim_array, [self.erry.ndim, self.erri.ndim])

        assert np.all(dim_array == 2), "Input arrays are not images! (i.e. dim != 2)"

        for i, shape in enumerate(shapes[1:], 1):
            assert shape == shapes[0], f"Shape mismatch: Array {i} has shape {shape}, expected {shapes[0]}"
        
        return None

    def _make_tab(self):
        ### Start applying angular slicing in each image
        aslice = {}
        
        if self.has_err:
            aslice["young"] = AngSlicing(self.imy, (self.xc, self.yc), mask=self.m1, err=self.erry,
                                         bin_width=self.bin_width, min_value=self.min_value)
            aslice["int"] = AngSlicing(self.imi, (self.xc, self.yc), mask=self.m2, err=self.erri,
                                       bin_width=self.bin_width, min_value=self.min_value)
        else:
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

    def _process_abin(self, i, tab):
        ### Get sub-table for a given angular bin
        tab_i = tab.loc[tab["bin"] == i].copy()
        tab_i.dropna(inplace=True)

        ### Sort sub-table by the distance (r) from the center
        tab_i = tab_i.sort_values("dr")
        tab_i.reset_index(drop=True, inplace=True)

        ### Calculate the total value within this particular angular slice
        if self.rlim is None: tot_flux = tab_i["values"].sum()
        else: tot_flux = tab_i["values"].loc[tab_i["dr"] <= self.rlim].sum()

        tab_i["cum_values"] = tab_i["values"].cumsum()     ### Cumulative sum
        tab_i["cum_frac"] = tab_i["cum_values"] / tot_flux ### Cumulative fraction

        ### Get list of tables, sliced at different Lag. radii                
        theta1 = i * self.bin_width
        theta2 = (i + 1) * self.bin_width
        mean_theta_rad = 0.5*(theta1 + theta2)

        if tab_i.empty:
            return (np.repeat(0., self.q.size), np.repeat(0., self.q.size),
                    np.repeat(0., self.q.size), np.repeat(0., self.q.size), None, None)
        else:
            cumfrac_values = tab_i["cum_frac"].clip(0., 1.) # Ensure there are no rounding problems
            rslice_i = np.interp(0.01 * self.q, cumfrac_values, tab_i["dr"])
            
            err = tab_i["values_err"].to_numpy() if self.has_err else None            
            rslice_chains = bootstrap_cumsum(tab_i, self.q, sig_y=err)

            xslice_i = (-1.) * rslice_i * np.sin(np.deg2rad(mean_theta_rad))
            yslice_i = (-1.) * rslice_i * np.cos(np.deg2rad(mean_theta_rad))

            if np.isin("dr_kpc", list(tab.columns)):
                rslice_phys_i = self.kpc_per_pixel * rslice_i
                
                rslice_phys_chains = {}
                for key in rslice_chains.keys():
                    rslice_phys_chains[key] = (self.kpc_per_pixel * rslice_chains[key][0],
                                               self.kpc_per_pixel * rslice_chains[key][1],
                                               self.kpc_per_pixel * rslice_chains[key][2])
            
            else: rslice_phys_i = np.zeros_like(rslice_i)

            return (xslice_i, yslice_i,
                    rslice_i, rslice_phys_i,
                    rslice_chains, rslice_phys_chains)
    
    def calc_lagradii(self, tab):
        ### Start by sorting table by angle value
        tab = tab.sort_values(by="theta", ascending=True)
        tab.reset_index(drop=True, inplace=True)

        ### Create the outputs
        shape = ((self.nbin, self.q.size))
        xslice = np.zeros(shape, dtype=np.float32)
        yslice, rslice, rslice_phys = np.zeros_like(xslice), np.zeros_like(xslice), np.zeros_like(xslice)
        rslice_err, rslice_phys_err = np.zeros_like(rslice), np.zeros_like(rslice_phys)

        ### Iterate over angle bins
        angle_bins = np.arange(0, self.nbin)
        if self.quiet:
            for i in angle_bins:
                abin_results = self._process_abin(i, tab)
                xslice[i, :], yslice[i, :], rslice[i, :], rslice_phys[i, :], chains_r, chains_rphys = abin_results
                
                if chains_r is not None:
                    rslice_err[i, :] = np.array([chains_r[key][1] for key in chains_r.keys()])
                    rslice_phys_err[i, :] = np.array([chains_rphys[key][1] for key in chains_rphys.keys()])
                else: pass
        else:
            for i in tqdm(angle_bins, desc="Iterating over angle bins", total=self.nbin):
                abin_results = self._process_abin(i, tab)
                xslice[i, :], yslice[i, :], rslice[i, :], rslice_phys[i, :], chains_r, chains_rphys = abin_results

                if chains_r is not None:
                    rslice_err[i, :] = np.array([chains_r[key][1] for key in chains_r.keys()])
                    rslice_phys_err[i, :] = np.array([chains_rphys[key][1] for key in chains_rphys.keys()])
                else: pass

        return {"x": xslice, "y": yslice,
                "r": rslice, "rphys": rslice_phys,
                "r_err": rslice_err, "rphys_err": rslice_phys_err}

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

