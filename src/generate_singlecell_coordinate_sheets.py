"""
Author: Myra Paz Masinas (Andrews and Boone Lab, 2024)
"""


from pathlib import Path
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="This script generates a CSV file containing single cell coordinates: "
                                                 "Image Path, Center_X and Center_Y for the purpose of labeling cells. "
                                                 "\nThis file can be used as input to the custom-made GUI single cell "
                                                 "labeling tool: https://github.com/BooneAndrewsLab/singlecelltool")
    parser.add_argument("-d", "--labeled-directory",
                        help="Directory containing labeled images (segmentation output).")
    parser.add_argument("-o", "--output-path",
                        help="Output path (please use absolute path and include the filename).")
    parser.add_argument("-p", "--index-image-path", type=int, default=0,
                        help="Index position of the target image path in the *_labeled_coords.npy output file. "
                             "The file includes the following metadata:\n"
                             "index 0 = Channel 1 (GFP marker) image path\n"
                             "index 1 = Channel 2 (Septin/Nuclear marker) image path\n"
                             "index 2 = Channel 3 (Cytoplasmic marker) image path\n"
                             "index 3 = center X coordinate of the cell\n"
                             "index 4 = center Y coordinate of the cell\n"
                             "The default is set to 0 (GFP marker).")
    parser.add_argument("-x", "--index-x", type=int, default=3,
                        help="Index position of the center X-coordinate of the cell in the segmentation "
                             "*_labeled_coords.npy output file. The default is set to 3.")
    parser.add_argument("-y", "--index-y", type=int, default=4,
                        help="Index position of the center Y-coordinate of the cell in the segmentation "
                             "*_labeled_coords.npy output file. The default is set to 4.")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_directory)
    output_path = args.output_path
    index_image_path = args.index_image_path
    index_x = args.index_x
    index_y = args.index_y

    # initialize data map
    data = {'Image Path': [], 'Center_X': [], 'Center_Y': []}

    # loop through all coordinate files in the input directory
    coordinate_files = sorted(labeled_dir.rglob("*labeled_coords.npy"))
    for coord_file in coordinate_files:
        print(f'Currently processing:{coord_file}')
        coord_data = np.load(coord_file)
        for cell in coord_data:
            data['Image Path'].append(cell[index_image_path])
            data['Center_X'].append(cell[index_x])
            data['Center_Y'].append(cell[index_y])

    # compile and save data to CSV
    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_path, index=False)
    print(f'Saved output file: {output_path}')

if __name__ == '__main__':
    main()
