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
from pathlib import Path
# Importing necessary programs from the libraries
from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio import features
from rasterstats import zonal_stats
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

#input and output directories
inDir = "./Input_Images"
outDir = "./Output_Images"

#save an image to the output directory from given array and name
def createIm(img: np.ndarray, name: str, parent: str):
    file_path = os.path.join(outDir, f"{parent}_{name}.jpg")
    cv2.imwrite(file_path, img)

#open image and return RGB tuple
def openIm(path: str, IMAGE_BITS: int =8):

    #check if the input file is a supported format
    red_channel = math.inf
    green_channel = math.inf
    blue_channel = math.inf

    #extract extension from file path
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension[1:].lower()
    if file_extension in {"jpg", "jpeg", "jpe", "jif", "jfif", "jfi"}:
        red_channel = 1
        green_channel = 2
        blue_channel = 3
    else:
        raise Exception(f"image file extension '{file_extension}' not currently supported")
    
    blue_raw: np.ndarray = None
    green_raw: np.ndarray = None
    red_raw: np.ndarray = None

    # importing the image to be analysed
    with rasterio.open(path, 'r') as raster2:
        raster2 = cast(rasterio.DatasetReader, raster2)
        band_count = cast(int, raster2.count)
        #confirm there are available bands for each channel
        if (band_count >= blue_channel):
            blue_raw = raster2.read(blue_channel)

        if (band_count >= green_channel):
            green_raw = raster2.read(green_channel)

        if (band_count >= red_channel):
            red_raw = raster2.read(red_channel)

    # Converting digital numbers (DN) to reflectance
    # This step is quite important and highly recommended, unless already have been done while image processing
    # Images converted to reflectance values should range between 0 - 1 or 0 - 100%
    # The equation for conversion is ((DN/(2^n)) - 1), where n is the bit size of the camera
    # Digital cameras generally store images as 8 bit or 16 bit
    # For this example n = 16, and thus ((DN/(2^16)) - 1) = 65535
    blue: np.ndarray
    green: np.ndarray
    red: np.ndarray

    image_max_value = (2 ** IMAGE_BITS) - 1

    if blue_raw is not None:
        blue = (cast(np.ndarray, blue_raw) / image_max_value).astype(float)

    if green_raw is not None:
        green = (cast(np.ndarray, green_raw) / image_max_value).astype(float)

    if red_raw is not None:
        red = (cast(np.ndarray, red_raw) / image_max_value).astype(float)


    return(blue, green, red)

def Exclude_Soil(Blue: np.ndarray, Green: np.ndarray, Red: np.ndarray):

    
    # Dealing with the situations division by zero
    np.seterr(divide='ignore', invalid='ignore')

    #This next section involves the calculation of Vegetation indices. 
    # Based on this article https://en.wikipedia.org/wiki/Vegetation_index#:~:text=A%20vegetation%20index%20(VI)%20is,activity%20and%20canopy%20structural%20variations.
    # A Vegetation Index is a calculation made using multispectral imagery data to highlight different qualities in vegetation such as dead plants or live leaves

    # Calculating vegetation indices
    # Making original calculation
    # Later on adjustments can be made to remove soil, or normalising the vegetation indices, etc.

    # These arrays give values between 0 and 1 but can also output negative values

    #These two VIs are RGB only. NGRDI normalises green against red
    NGRDI_Orig = ((Green).astype(float) - (Red).astype(float))/((Green).astype(float) + (Red).astype(float))
    #ExGI (Excess green index) cotrasts green against the red and blue bands
    ExGI_Orig = ((2*(Green).astype(float)) - ((Red).astype(float) + (Blue).astype(float)))

    #output the NGRDI Vegetation index masked by the ExGI index to remove soil
    NGRDI = np.where(ExGI_Orig > 0.05, NGRDI_Orig*255, -math.inf)

    return NGRDI

def countContours(img_path: str):
    print(os.path.join(outDir, f"{img_path}.jpg"))
    #read in the soil excluded image
    img_vals = cv2.imread(os.path.join(outDir, f"{img_path}.jpg"))
    #ensure the image is greyscale
    imgrey = cv2.cvtColor(img_vals, cv2.COLOR_BGR2GRAY)
    #convert the greyscale image to binary
    ret, thresh  = cv2.threshold(imgrey, 10, 255, cv2.THRESH_BINARY)
    #define kernels for eorsion/dilation
    kernel1= np.ones((4,4), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)

    #erode image to remove noise and dilate to restore larger shapes
    eroded = cv2.erode(thresh, kernel1, iterations = 2)
    dilated = cv2.dilate(eroded, kernel2, iterations = 8)

    #find contours in the image
    contours, heirachy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    count = len(contours)
    
    #outline the found contours on the dilated image
    cv2.drawContours(dilated, contours, -1, (225, 255, 0), 5)

    return(count, dilated)

#perform plant count for each input file

if(os.path.exists(inDir)):
    files = 0
    for file in os.listdir(inDir):
        filename = os.fsdecode(file)
        b,g,r = openIm(os.path.join(inDir, filename))

        ngrdi = Exclude_Soil(b, g, r)
        createIm(ngrdi, "Masked_NGRDI", Path(filename).stem)


        count, contours = countContours(Path(filename).stem + "_" + "Masked_NGRDI")
        createIm(contours, "contours", Path(filename).stem)

        print (f"{filename}: contains {count} plants")
        files += 1
    print(f"{files} files analysed")
else:
    print(f"Unable to find {inDir} file path.")