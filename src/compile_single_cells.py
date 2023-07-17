"""
Author: Oren Kraus (https://github.com/okraus, 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, 2023)
"""


from skimage.measure import regionprops
from skimage.io import imread as skimage_imread
from PIL import Image, ImageFilter, ImageEnhance
import skimage
import scipy.ndimage as nd
import mahotas as mh
import numpy as np
import os


def not_on_border(width, height, loc_left, loc_upper, loc_right, loc_lower):
    if loc_left > 0 and loc_right < width and loc_upper > 0 and loc_lower < height:
        return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label_directory", help="Path containing labeled images (segmentation output)")
    parser.add_argument("-i", "--image_directory", help="Path containing input microscopy images")
    parser.add_argument("-s", "--crop_size", type=int, default=64, help="Single cell crop size. Default is 64.")
    parser.add_argument("-g", "--gfp_channel", default="ch1",
                        help="Channel where the GFP (Green Fluorescent Protein) marker is. Default is: ch1")
    parser.add_argument("-n", "--nuclear_channel", default="ch2",
                        help="Channel to be used in segmentation - usually where the nuclear and/or septin "
                             "markers are. Default is: ch2")
    parser.add_argument("-c", "--cyto_channel", default="ch3",
                        help="Channel where the cytoplasmic marker is. Default is: ch3")
    args = parser.parse_args()

    labeldir = args.label_directory
    imdir = args.image_directory
    gfp_channel = args.gfp_channel
    nuclear_channel = args.nuclear_channel
    cyto_channel = args.cyto_channel
    crop_size = args.crop_size
    s = crop_size / 2

    se5 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=int)
    se3 = nd.generate_binary_structure(2, 1)

    labels = sorted(filter(lambda x: x.endswith('_labeled.tiff'), os.listdir(labeldir)))
    len_labels = len(labels)

    final_cellcnt = 0
    for i, label in enumerate(labels):
        # prepare image paths
        well = label.split('_')[0]
        label_in = os.path.join(labeldir, label)
        imm_in = label_in.replace('labeled.tiff', 'imm.tiff')
        label_out = label_in.replace('.tiff', '.npy')
        coords_out = label_in.replace('.tiff', '_coords.npy')

        gfp_in = os.path.join(imdir, '%s-%ssk1fk1fl1.tiff' % (well, gfp_channel))
        sepnuc_in = gfp_in.replace(gfp_channel, nuclear_channel)
        cyto_in = gfp_in.replace(gfp_channel, cyto_channel)

        # read images
        labeled = skimage_imread(label_in)
        imm = skimage_imread(imm_in)
        gfp = skimage_imread(gfp_in)
        sepnuc = skimage_imread(sepnuc_in)
        cyto = skimage_imread(cyto_in)

        # brighten sepnuc channel
        sepnuc_gamma = skimage.exposure.adjust_gamma(sepnuc, gamma=1.25)
        im_arr = sepnuc_gamma.astype(float)
        im_scale = 1/im_arr.max()
        sepnuc_new = ((im_arr*im_scale)*255).round().astype(np.uint8)
        im_bright = Image.fromarray(sepnuc_new)
        sepnuc_bright = ImageEnhance.Brightness(im_bright).enhance(3)
        im_con = ImageEnhance.Contrast(sepnuc_bright)
        sepnuc_enhanced = np.array(im_con.enhance(2))

        # brighten cyto channel
        cyto_gamma = skimage.exposure.adjust_gamma(cyto, gamma=2.25)
        im_arr = cyto_gamma.astype(float)
        im_scale = 1/im_arr.max()
        cyto_new = ((im_arr*im_scale)*255).round().astype(np.uint8)
        im_bright = Image.fromarray(cyto_new)
        cyto_bright = ImageEnhance.Brightness(im_bright).enhance(1)
        im_con = ImageEnhance.Contrast(cyto_bright)
        cyto_enhanced = np.array(im_con.enhance(1.5))

        # filter out small and big labels
        labeled, c = mh.labeled.filter_labeled(labeled, remove_bordering=True, min_size=200, max_size=6000)
        width, height = labeled.shape

        # prepare cells into array
        cells_coords = []
        all_cells = []
        measurements = regionprops(labeled)
        cellcnt = len(measurements)
        for ic in range(cellcnt):
            label_val = measurements[ic].label
            centroid = measurements[ic].centroid
            center_x = centroid[1]
            center_y = centroid[0]
            loc_left = int(center_x - s)
            loc_upper = int(center_y - s)
            loc_right = int(center_x + s)
            loc_lower = int(center_y + s)
            if not_on_border(width, height, loc_left, loc_upper, loc_right, loc_lower):
    #             print(ic, center_x, center_y)
                cells_coords.append([center_x, center_y])
                cell_cropped = labeled[loc_upper:loc_lower, loc_left:loc_right]
                cell_cropped_mask = np.where(cell_cropped == label_val, 1, 0)

                # NEW!! - smooth the label edges then dilate
                cell_cropped_img = Image.fromarray(cell_cropped_mask.astype('uint8'), mode='L')
                cell_croppped_filtered  = cell_cropped_img.filter(ImageFilter.ModeFilter(size=3))
                new_cell_cropped_mask = np.array(cell_croppped_filtered)
                cell_cropped_dilate = new_cell_cropped_mask
                final_cell_cropped_mask = cell_cropped_dilate

                # get imm, gfp, sepnuc, and cyto crops
                imm_cropped = imm[loc_upper:loc_lower, loc_left:loc_right]
                gfp_cropped = gfp[loc_upper:loc_lower, loc_left:loc_right]
                sepnuc_cropped = sepnuc_enhanced[loc_upper:loc_lower, loc_left:loc_right]
                cyto_cropped = cyto_enhanced[loc_upper:loc_lower, loc_left:loc_right]

                imm_cropped_masked = np.where(final_cell_cropped_mask==1, imm_cropped, 0)

                # get masked crops
                gfp_cropped_masked = np.where(final_cell_cropped_mask==1, gfp_cropped, 0)
                sepnuc_cropped_masked = np.where(final_cell_cropped_mask==1, sepnuc_cropped, 0)
                cyto_cropped_masked = np.where(final_cell_cropped_mask==1, cyto_cropped, 0)

                cell_stacked = np.ravel([sepnuc_cropped_masked, gfp_cropped_masked,
                                         cyto_cropped_masked, imm_cropped_masked, cyto_cropped])
                all_cells.append(cell_stacked)

        cur_cellcount = len(cells_coords)
        final_cellcnt += cur_cellcount

        if cur_cellcount > 0:
            cells_array = np.stack(all_cells)
            cells_coords = np.array(cells_coords)

            # save npy files
            np.save(label_out, cells_array)
            np.save(coords_out, cells_coords)

if __name__ == '__main__':
    main()
