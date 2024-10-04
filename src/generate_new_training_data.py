"""
Author: Myra Paz Masinas (Andrews and Boone Lab, 2024)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import h5py


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="This script generates the train and test HDF5 files needed to train "
                                                 "a new CycleNET network (https://github.com/BooneAndrewsLab/CycleNET)")
    parser.add_argument("-d", "--labeled-directory",
                        help="Directory containing labeled images (segmentation output).")
    parser.add_argument("-i", "--input-file",
                        help="File that contains the labeled cells. Required columns should be in the order of: "
                             "Image Path, Center_X, Center_Y, Label")
    parser.add_argument("-t", "--train-file",
                        help="Output path for the training set (please use absolute path and include the filename)")
    parser.add_argument("-v", "--test-file",
                        help="Output path for the test set (please use absolute path and include the filename)")
    parser.add_argument("-x", "--index-x", type=int, default=3,
                        help="Index position of the center X-coordinate of the cell in the segmentation "
                             "*_labeled_coords.npy output file. The default is set to 3. "
                             "The file includes the following metadata:\n"
                             "index 0 = Channel 1 (GFP marker) image path\n"
                             "index 1 = Channel 2 (Septin/Nuclear marker) image path\n"
                             "index 2 = Channel 3 (Cytoplasmic marker) image path\n"
                             "index 3 = center X coordinate of the cell\n"
                             "index 4 = center Y coordinate of the cell\n"
                        )
    parser.add_argument("-y", "--index-y", type=int, default=4,
                        help="Index position of the center Y-coordinate of the cell in the segmentation "
                             "*_labeled_coords.npy output file. Please see index details on the -x argument help text. "
                             "The default is set to 4.")
    parser.add_argument("-c", "--labels-cellcycle", action='store_true',
                        help="Use this flag if the labels are cell cycle phases. Default is False.")
    parser.add_argument("-l", "--labels-localization", action='store_true',
                        help="Use this flag if the labels are protein localization. Default is False.")
    parser.add_argument("-s", "--crop-size", type=int, default=64,
                        help="Single cell crop size. Default is 64.")
    parser.add_argument("-n", "--channel", type=int, default=5,
                        help="Number of channels/frames saved in the segmentation output *_labeled.npy file. "
                             "Default is 5.")
    parser.add_argument("-m", "--metadata", type=int, default=4,
                        help="Number of metadata to include in the output file. Default is 4.")
    parser.add_argument("-w", "--image-width", type=int, default=1339,
                        help="Image width. Default is 1339.")
    parser.add_argument("-z", "--image-height", type=int, default=1001,
                        help="Image height. Default is 1001.")
    parser.add_argument("-r", "--split-ratio", type=float, default=0.8,
                        help="Ratio to use when splitting the train and test datasets. Default is 0.8")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_directory)
    input_path = Path(args.input_file)
    output_train = Path(args.train_file)
    output_test = Path(args.test_file)
    ix_loc_x = args.index_x
    ix_loc_y = args.index_y
    crop_size = args.crop_size
    num_chan = args.channel
    num_info = args.metadata
    w = args.image_width
    h = args.image_height
    split_ratio = args.split_ratio

    # Set class list
    labels_cellcycle = args.labels_cellcycle
    labels_localization = args.labels_localization
    if not any([labels_cellcycle, labels_localization]):
        raise ValueError("At least one of the following is a required flag to set the class list: "
                         "-c (cell cycle) or -l (localization)")
    if all([labels_cellcycle, labels_localization]):
        raise ValueError("Please use only one flag from the following class list options: "
                         "-c (cell cycle) or -l (localization)")
    if labels_cellcycle:
        classes = ['Early G1', 'Late G1', 'S/G2', 'Metaphase', 'Anaphase', 'Telophase',
                   'Abberent', 'Over_seg', 'Anaphase_defect']
    if labels_localization:
        classes = ['Actin', 'Bud', 'Bud Neck', 'Bud Periphery', 'Bud Site', 'Cell Periphery', 'Cytoplasm',
                   'Cytoplasmic Foci', 'Eisosomes', 'Endoplasmic Reticulum',  'Endosome', 'Golgi',
                   'Lipid Particles', 'Mitochondria', 'None', 'Nuclear Periphery', 'Nucleolus', 'Nucleus',
                   'Peroxisomes', 'Punctate Nuclear', 'Vacuole', 'Vacuole Periphery']
    num_classes = len(classes)

    # Load the labeled file
    df = pd.read_csv(input_path)
    maxcells = df.shape[0]

    # Initialize output arrays
    output = np.zeros((maxcells, crop_size ** 2 * num_chan), dtype=np.uint16)  # array of cropped cells (flattened)
    classlabels = np.zeros((maxcells, num_classes), dtype=np.int8)  # class labels in one-hot encoding
    cellinfo = np.chararray((maxcells, num_info), itemsize=200)

    # Loop through each labeled cell
    labeled_images = {}
    cell_index = 0
    for image_path, x, y, loc_label in df.itertuples(index=False):
        well = image_path.split('/')[-1][:12]
        if image_path not in labeled_images:
            print(f'Loading cell data for {image_path}')
            labeled_array = np.load(labeled_dir / f'{well}_labeled.npy')
            cell_count = labeled_array.shape[0]
            cells = labeled_array.reshape(cell_count, num_chan, crop_size, crop_size)

            coords_array = np.load(labeled_dir / f'{well}_labeled_coords.npy')
            coords_data = [(int(float(c[ix_loc_x])), int(float(c[ix_loc_y]))) for c in coords_array]

            labeled_images[image_path] = {}
            for i, cell_coord in enumerate(coords_data):
                labeled_images[image_path][cell_coord] = cells[i]

        # only process non-border cells
        if (x - crop_size / 2) > 0 and (x + crop_size / 2) < w and (y - crop_size / 2) > 0 and (y + crop_size / 2) < h:
            cell_crops = labeled_images[image_path][cell_coord].ravel()
            output[cell_index] = cell_crops
            classlabels[cell_index, classes.index(loc_label)] = 1
            cellinfo[cell_index] = [image_path, well, x, y]
            cell_index += 1

    # Reshape output arrays to include non-border cells only
    output = output[:cell_index]
    classlabels = classlabels[:cell_index]
    cellinfo = cellinfo[:cell_index]
    print(f'total processed cells: {cell_index} out of {maxcells}')

    # Randomize output arrays
    randind = list(range(len(output)))
    np.random.shuffle(randind)
    randoutput = output[randind]
    randclasses = classlabels[randind]
    randinfo = cellinfo[randind]

    # Split datasets into train and test sets
    splitratio = split_ratio
    totalcells = len(randoutput)

    traincells = randoutput[:int(totalcells * splitratio)]
    trainlabels = randclasses[:int(totalcells * splitratio)]
    traininfo = randinfo[:int(totalcells * splitratio)]

    testcells = randoutput[int(totalcells * splitratio):]
    testlabels = randclasses[int(totalcells * splitratio):]
    testinfo = randinfo[int(totalcells * splitratio):]

    print(f'train set: {traincells.shape}')
    print(f'test set: {testcells.shape}')

    # Save train set
    info_cols = np.array(['Image Path', 'Well', 'Center_X', 'Center_Y']).astype('|S')

    if output_train.exists():
        output_train.unlink()

    train_file = h5py.File(output_train, 'w')
    train_file['data1'] = traincells
    train_file['Index1'] = trainlabels
    train_file['Index1'].attrs['columns'] = np.array(classes).astype('|S')
    train_file['Info1'] = traininfo
    train_file['Info1'].attrs['columns'] = info_cols
    train_file.close()

    # Save train set
    info_cols = np.array(['Image Path', 'Well', 'Center_X', 'Center_Y']).astype('|S')

    if output_test.exists():
        output_test.unlink()

    test_file = h5py.File(output_test, 'w')
    test_file['data1'] = testcells
    test_file['Index1'] = testlabels
    test_file['Index1'].attrs['columns'] = np.array(classes).astype('|S')
    test_file['Info1'] = testinfo
    test_file['Info1'].attrs['columns'] = info_cols
    test_file.close()

    print("Done generating new training and test datasets.")

if __name__ == '__main__':
    main()