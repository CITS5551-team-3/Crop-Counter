# Importing necessary libraries
from typing import cast
import geopandas as gpd
import rasterio.mask
import fiona
import os
import numpy as np
import shapefile
import rasterstats
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import rasterio
import time
import cv2
import math
# Importing necessary programs from the libraries
from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio import features
from rasterstats import zonal_stats
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


# clipping image based on a shape file is not necessary for initial tests since the provided images are smaller than a single plot. 
# This section may be added later once crop data has been collected

# importing the image to be analysed
with rasterio.open("Test_Images/Img_RGB.jpg", 'r') as raster2:
    raster2 = cast(rasterio.DatasetReader, raster2)

    bandCount = cast(int, raster2.count)
    if (bandCount >= 1):
        Blue16bit = raster2.read(1) # 1 stands for Blue band 
    if (bandCount >= 2):
        Green16bit = raster2.read(2) # 2 stands for Green band 
    if (bandCount >= 3):
        Red16bit = raster2.read(3) # 3 stands for Red band
    if (bandCount >= 4):
        NIR16bit = raster2.read(4) # 4 stands for Red band
    if (bandCount >= 5):
        RE16bit = raster2.read(5) # 5 stands for Red band

# Converting digital numbers (DN) to reflectance
# This step is quite important and highly recommended, unless already have been done while image processing
# Images converted to reflectance values should range between 0 - 1 or 0 - 100%
# The equation for conversion is ((DN/(2^n)) - 1), where n is the bit size of the camera
# Digital cameras generally store images as 8 bit or 16 bit
# For this example n = 16, and thus ((DN/(2^16)) - 1) = 65535

# Initialise empty array so the type checker doesn't think the variables are unbound
Blue = np.array([])
Green = np.array([])
Red = np.array([])
NIR = np.array([])
RE = np.array([])

if (bandCount >= 1):
    Blue = (cast(np.ndarray, Blue16bit) / 255).astype(float)
if (bandCount >= 2):
    Green = (cast(np.ndarray, Green16bit) / 255).astype(float)
if (bandCount >= 3):
    Red = (cast(np.ndarray, Red16bit) / 255).astype(float)
if (bandCount >= 4):
    NIR = (cast(np.ndarray, NIR16bit) / 255).astype(float)
if (bandCount >= 5):
    RE = (cast(np.ndarray, RE16bit) / 255).astype(float)

# This next section involves the calculation of Vegetation indices. 
# Based on this article https://en.wikipedia.org/wiki/Vegetation_index#:~:text=A%20vegetation%20index%20(VI)%20is,activity%20and%20canopy%20structural%20variations.
# A Vegetation Index is a calculation made using multispectral imagery data to highlight different qualities in vegetation such as dead plants or live leaves

# Calculating vegetation indices
# This example shows with 15 vegetation indices
# Any number of vegetation indices can be used

# Dealing with the situations division by zero
np.seterr(divide='ignore', invalid='ignore')

# Making original calculation
# Later on adjustments can be made to remove soil, or normalising the vegetation indices, etc.

# These arrays give values between 0 and 1 but can also output negative values

# RGB only
if (bandCount >= 3):
    NGRDI_Orig = ((Green).astype(float) - (Red).astype(float))/((Green).astype(float) + (Red).astype(float))
    HUE = np.arctan((2 * (Red.astype(float)  - Green.astype(float)  - Blue.astype(float) ) )/ (30.5*(Green.astype(float)  - Blue.astype(float) )))

# NIR based
if (bandCount >= 4):
    NDVI_Orig = (NIR.astype(float) - Red.astype(float)) / (NIR.astype(float) + Red.astype(float))
    GNDVI_Orig = (NIR.astype(float) - Green.astype(float)) / (NIR.astype(float) + Green.astype(float))
    ENDVI_Orig = (NIR.astype(float) + Green.astype(float) - 2*Blue.astype(float)) / (NIR.astype(float) + Green.astype(float) + 2*Blue.astype(float))
    SIPI_Orig = (NIR.astype(float)-Blue.astype(float))/(NIR.astype(float) + Red.astype(float))
    NLI_Orig = (((NIR.astype(float))**2) - Red.astype(float)) / (((NIR.astype(float))**2) + Red.astype(float))
    SR_Orig = NIR.astype(float)/Red.astype(float)
    DVI_Orig = NIR.astype(float) - Red.astype(float)
    RDVI_Orig = (NIR.astype(float) - Red.astype(float)) / ((NIR.astype(float) + Red.astype(float))**(1/2))

# RE based
if (bandCount >= 5):
    RENDVI_Orig = (RE.astype(float) - Red.astype(float)) / (RE.astype(float) + Red.astype(float))
    NDRE_Orig = (NIR.astype(float) - RE.astype(float)) / (NIR.astype(float) + RE.astype(float))
    NNIR_Orig = NIR.astype(float) / (NIR.astype(float) + (RE.astype(float) + Green.astype(float)))
    MCARI_Orig = (RE.astype(float)-Red.astype(float)) - 2*(RE.astype(float) - Green.astype(float))*(RE.astype(float) / Red.astype(float))
    MDD_Orig = (NIR.astype(float) - RE.astype(float)) - (RE.astype(float) - Green.astype(float))
    MARI_Orig = ((1/Green.astype(float))-(1/RE.astype(float)))*NIR.astype(float)

print(bandCount)

if bandCount >= 4:
    # these images scale the 0 - 1 values to a 255 greyscale
    # sanity check to make sure the Vegetation Indices are producing images:
    img_vals = (cast(np.ndarray, NDVI_Orig) * 255).astype(int)
    cv2.imwrite("Test_Images/VI_Test.jpg", img_vals)
    # sanity check to make sure the Vegetation Indices are producing images:
    img_vals = (cast(np.ndarray, ENDVI_Orig) * 255).astype(int)
    cv2.imwrite("Test_Images/VI_Test_2.jpg", img_vals)
    # sanity check to make sure the Vegetation Indices are producing images:
    img_vals = (cast(np.ndarray, SIPI_Orig) * 255).astype(int)
    cv2.imwrite("Test_Images/VI_Test_3.jpg", img_vals)

# Separating crop and soil fractions based on NGRDI
# This step is optional and based on the need of the research
# For this example NGRDI has been used to classify between soil and crop, so anywhere in the image that aligns with NGRDI being negative  will be removed from the selected index
# Any vegetation indices can be used (VI_For_Classification)
# basic syntax is: 
# VI = np.where(VI_For_Classification (symbol(s): > or < or = or !=) (classification criteria), VI_Of_Interest, -9999)
# -9999 is the number for null values

# these arrays contain values ranging from 0 - 1. GNDVI has been tweaked following some experimentation
# with the test data found (Test_Image_8) GNDVI seems particularly good at isolating the plants, but with a value of 0 showing the plant and 1 showing shadow
# GNDVI also seem to be best in this image for isolating or ignoring shadows. the threshold has also been adjusted based on observation
# 0.15 removes a good amount of the soil and less green crops
hue_vals = ((cast(np.ndarray, HUE) - (math.pi/2))/ -math.pi *255).astype(int)

# RGB only
if (bandCount >= 3):
    NGRDI = np.where(NGRDI_Orig > 0.15, NGRDI_Orig, -9999)
    hue_excl = np.where(NGRDI_Orig > 0.15, hue_vals, -math.inf)

# NIR based
if (bandCount >= 4):
    NDVI = np.where(NGRDI_Orig > 0, NDVI_Orig, -9999)
    GNDVI = np.where(NGRDI_Orig > 0.15, (1-GNDVI_Orig), -9999)
    ENDVI = np.where(NGRDI_Orig > 0, ENDVI_Orig, -9999)
    SIPI = np.where(NGRDI_Orig > 0, SIPI_Orig, -9999)
    NLI = np.where(NGRDI_Orig > 0, NLI_Orig, -9999)
    SR = np.where(NGRDI_Orig > 0, SR_Orig, -9999)
    DVI = np.where(NGRDI_Orig > 0, DVI_Orig, -9999)
    RDVI = np.where(NGRDI_Orig > 0, RDVI_Orig, -9999)

# RE based
if (bandCount >= 5):
    RENDVI = np.where(NGRDI_Orig > 0, RENDVI_Orig, -9999)
    NDRE = np.where(NGRDI_Orig > 0, NDRE_Orig, -9999)
    NNIR = np.where(NGRDI_Orig > 0, NNIR_Orig, -9999)
    MCARI = np.where(NGRDI_Orig > 0, MCARI_Orig, -9999)
    MDD = np.where(NGRDI_Orig > 0, MDD_Orig, -9999)
    MARI = np.where(NGRDI_Orig > 0, MARI_Orig, -9999)

# this image scale the 0 - 1 values to a 255 greyscale
# sanity check to make sure the soil exclusion is working, currently no change since NGRDI_Orig already has been classified by itself:
#img_vals = ((cast(np.ndarray, hue_excl) + (math.pi/2))/ math.pi *255).astype(int)
cv2.imwrite("Test_Images/Hue_Exclusiont.jpg", hue_excl)
cv2.imwrite("Test_Images/NGRDI.jpg", NGRDI*255)
