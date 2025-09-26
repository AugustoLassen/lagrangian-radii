# CHANGELOG.md
*By A. Lassen & R. Smith*  
*Last updated: September 26, 2025*

------------------- Change log  
**26/09/25**
- Code version 0.3.1 --> 0.4
-- Added support for error arrays
-- Implemented bootstrap_cumsum() to estimate uncertainties of the Lagrangian radii
-- Refactored _process_abin() to derive Lagrangian radii using interpolation on the cumulative distribution within each angular bin
-- calc_lagradii() now also returns uncertainties (r_err, rphys_err) along with radii
-- Deprecated fill_zeros() function
-- Improve validation process of input arrays and dictionaries

**29/07/25**  
- Modularization of `calc_lagradii()` method  
- Added `fill_zeros()` utility function for error handling for edge cases in angular bin processing  

**24/07/25**  
- Updated methods for inclination correction  

**15/07/25**  
- SSD calculation has been correctly implemented into class  

**14/07/25**  
- Fixed incorrect calculation of kpc/pixel parameters  
