"""
Author: Oren Kraus (https://github.com/okraus, 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, 2023)
"""

from skimage.io import imread as skimage_imread
from skimage.io import imsave as skimage_imsave
import scipy.ndimage as nd
import mahotas as mh
import numpy as np
import sys
import os


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="Path to input folder containing images to be segmented")
    parser.add_argument("-o", "--output_folder", help="Path to output folder where to save labeled images")
    parser.add_argument("-s", "--scripts_folder", help="Path where the scripts are saved")
    parser.add_argument("-g", "--gfp_channel", help="Channel where the GFP (Green Fluorescent Protein) marker is. Example: ch1")
    parser.add_argument("-n", "--nuclear_channel", help="Channel to be used in segmentation - usually where the nuclear and/or septin markers are. Example: ch2")
    parser.add_argument("-c", "--cyto_channel", help="Channel where the cytoplasmic marker is. Example: ch3")
    args = parser.parse_args()
    
    indir = args.input_folder
    outdir = args.output_folder
    scriptdir = args.scripts_folder
    gfp_channel = args.gfp_channel
    nuclear_channel = args.nuclear_channel
    cyto_channel = args.cyto_channel

    sys.path.insert(0, scriptdir)

    import NSMM
    from Watershed_MRF import Watershed_MRF

    files = sorted(filter(lambda x: gfp_channel in x, os.listdir(indir)))
    for filename in files:
        filepath = os.path.join(indir, filename)
        outpath = os.path.join(outdir, "%s_labeled.tiff" % filename.split('-')[0])
        outpath_imm = outpath.replace("labeled", "imm")
        if os.path.isfile(outpath) and os.path.isfile(outpath_imm):
          print("Already processed %s" % filename)
          continue
        else:
          print("Processing %s" % filename)
        #image_green = skimage_imread(filepath) # GFP - not needed in this script
        image_red = skimage_imread(filepath.replace(gfp_channel, nuclear_channel)) # Nuclear and Septin marker
        image_farred = skimage_imread(filepath.replace(gfp_channel, cyto_channel)) # Cytoplasmic marker
        
        # https://github.com/okraus/cell_segmentation_2_github
        # NSMM + Watershed
        try:
          I_MM, I_MM_Sep = NSMM.I_MM_BEN(image_farred, image_red)
          LabelIm = Watershed_MRF(image_farred, I_MM)
          
          Sep_Lab, Sep_num = nd.measurements.label(I_MM_Sep == 1)
          Changed_cells = np.array([], dtype=np.uint8)
          
          for Lab in range(Sep_num):
              UniqueCells = np.unique(LabelIm[Sep_Lab == Lab])
              UniqueCells = UniqueCells[UniqueCells > 1]
              if len(UniqueCells) > 1:
                  sumCellNum = np.zeros(len(UniqueCells), dtype=int)
                  for j in range(len(UniqueCells)):
                      sumCellNum[j] = np.sum(LabelIm[Sep_Lab == Lab] == UniqueCells[j])
                  SortedUniqueCells = UniqueCells[np.argsort(sumCellNum)]
                  if not np.sum(Changed_cells == SortedUniqueCells[-1]):
                      LabelIm[LabelIm == SortedUniqueCells[-2]] = SortedUniqueCells[-1]
                      Changed_cells = np.append(Changed_cells, SortedUniqueCells[-1])
                      
          LabelIm = mh.labeled.remove_bordering(LabelIm)
          skimage_imsave(outpath, LabelIm.astype(np.int16))
          skimage_imsave(outpath_imm, I_MM.astype(np.int16))
        except:
          print("\tError found. Skipping %s" % filepath) 


if __name__ == '__main__':
    main()

