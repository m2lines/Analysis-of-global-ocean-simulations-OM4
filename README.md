This repository is a tool which can be used for **fast** analysis of multiple global ocean simulations OM4:
![](https://github.com/m2lines/Analysis-of-global-ocean-simulations-OM4/blob/master/assets/Preview.png)

* See how to compare simulations according to available metrics in [notebooks-core/00-analysis.ipynb](https://github.com/m2lines/Analysis-of-global-ocean-simulations-OM4/blob/master/notebooks-core/00-analysis.ipynb).
* Examples of usage of the package in research can be found in [notebooks-analysis](https://github.com/m2lines/Analysis-of-global-ocean-simulations-OM4/tree/master/notebooks-analysis).

For full functionality, download WOA18 temperature and salinity to `data` folder:
```
cd data
wget https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav81B0/1.00/woa18_decav81B0_t00_01.nc woa_1981_2010.nc
wget https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav81B0/1.00/woa18_decav81B0_s00_01.nc
```
