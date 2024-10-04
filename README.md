# CycleNET
[![DOI](https://zenodo.org/badge/658895858.svg)](https://zenodo.org/doi/10.5281/zenodo.10998620)

### Prerequisites and Major Dependencies
* Anaconda
* Python 3.6
* Tensorflow 1.6


---
### Project Setup

Clone the repository
```
$ git clone https://github.com/BooneAndrewsLab/CycleNET.git
$ cd CycleNET
```

Download [Cell Cycle and Localization networks][cellcyclenet_models] to 'models' folder
```
$ mkdir models
$ wget -P models https://thecellvision.org/cellcycleomics/cellcyclenet_models.tar.gz
$ tar -xvzf models/cellcyclenet_models.tar.gz --directory=models
```

Download [Cell Cycle][cellcycle_training_dataset] and [Localization][localization_training_dataset] training sets to 'datasets' folder
```
$ mkdir datasets
$ wget -P datasets https://thecellvision.org/cellcycleomics/cellcycle_training_dataset.tar.gz
$ wget -P datasets https://thecellvision.org/cellcycleomics/localization_training_dataset.tar.gz
$ tar -xvzf datasets/cellcycle_training_dataset.tar.gz --directory=datasets
$ tar -xvzf datasets/localization_training_dataset.tar.gz --directory=datasets
```

Create a conda environment
```
$ conda env create -f environment.yml
$ conda activate cyclenet_env
```

---

### Training the Cell Cycle and Localization Networks

#### CELL CYCLE

```
Usage
$ mkdir <MODEL_OUTPUT_FOLDER>
$ python src/training_script_cellcycle.py -i <INFERENCE_FUNCTION> -l <MODEL_OUTPUT_FOLDER> -t <TRAINING_SET_FILE> -v <TEST_SET_FILE>

Example:
$ mkdir model_training_cellcycle
$ python src/training_script_cellcycle.py -i inference_oren -l model_training_cellcycle -t datasets/cellcycle_train_set.hdf5 -v datasets/cellcycle_test_set.hdf5
```

#### LOCALIZATION

```
Usage
$ mkdir <MODEL_OUTPUT_FOLDER>
$ python src/training_script_localization.py -i <INFERENCE_FUNCTION> -l <MODEL_OUTPUT_FOLDER> -t <TRAINING_SET_FILE> -v <TEST_SET_FILE>

Example:
$ mkdir model_training_localization
$ python src/training_script_localization.py -i inference_leo -l model_training_localization -t datasets/localization_train_set.hdf5 -v datasets/localization_test_set.hdf5
```

---

### Pipeline: Single Cell Segmentation + Cell Cycle and Localization Prediction

#### STEP 1 - SINGLE CELL SEGMENTATION
```
Usage: 
$ python src/segmentation.py -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -s <SCRIPTS_FOLDER> -g <GFP_CHANNEL> -n <NUCLEAR_CHANNEL> -c <CYTO_CHANNEL>

Example:
$ python src/segmentation.py -i example/input_images -o example/labeled_images -s ./src -g ch1 -n ch2 -c ch3
```

Script parameters:
```
  -i INPUT_FOLDER       Path to input folder containing images to be segmented
  -o OUTPUT_FOLDER      Path to output folder where to save labeled images
  -s SCRIPTS_FOLDER     Path where the scripts are saved
  -g GFP_CHANNEL        Channel where the GFP (Green Fluorescent Protein) marker is. Example: ch1
  -n NUCLEAR_CHANNEL    Channel to be used in segmentation - usually where the nuclear and/or septin markers are. Example: ch2
  -c CYTO_CHANNEL       Channel where the cytoplasmic marker is. Example: ch3
```

_This script calls src/**NSMM**.py and src/**Watershed_MRF**.py_


#### STEP 2 - COMPILE SINGLE CELL CROPS AND COORDINATES
```
Usage: 
$ python src/compile_single_cells.py -l <LABELED_FOLDER> -i <IMAGE_FOLDER> -s <CROP_SIZE> -g <GFP_CHANNEL> -n <NUCLEAR_CHANNEL> -c <CYTO_CHANNEL>

Example:
$ python src/compile_single_cells.py -l example/labeled_images -i example/input_images -s 64 -g ch1 -n ch2 -c ch3
```

#### STEP 3 - CELL CYCLE AND LOCALIZATION PREDICTION

```
Usage:
$ python src/evaluation_script_localization_cellcycle.py -l <LOC_CPKT> -c <CYC_CPKT> -i <INPUT_PATH> -o <OUTPATH> -n

Example:
$ python src/evaluation_script_localization_cellcycle.py -l models/localization/localization.ckpt-6500 -c models/cellcycle/cell_cycle.ckpt-9500 -i example/labeled_images -o example/predictions
```

Script parameters:
```
  -l LOC_CPKT           Path to model/checkpoint for localization network
  -c CYC_CPKT           Path to model/checkpoint for cell cycle network
  -s INPATH             Path to input folder containing labeled images
  -o OUTPATH            Where to store output csv files
```
_This script calls src/**preprocess_images**.py and src/**input_queue_whole_screen**.py_

---

### Generating New Training Dataset
Users can generate their own train and test HDF5 datasets to train the model from scratch. The process requires steps 
1 and 2 from the pipeline described above.

#### STEP 1 - SINGLE CELL SEGMENTATION and COMPILE SINGLE CELL CROPS AND COORDINATES
Please see instructions on how to run the segmentation and crops compilation from steps 1 and 2 of above 
pipeline. The output of these two methods are the required input of the succeeding steps.

#### STEP 2 - GENERATE COORDINATE SHEET
This script generates a CSV file containing single cell coordinates Image Path, Center_X and Center_Y for the purpose 
of labeling cells. 

```
Usage:
$ python src/generate_singlecell_coordinate_sheets.py -d <LABELED_DIRECTORY> -o <OUTPUT_PATH> -p <INDEX_IMAGE_PATH> -x <INDEX_X> -y -<INDEX_Y>

Example (cell cycle):
$ python src/generate_singlecell_coordinate_sheets.py -d example/labeled_images -o /home/username/git/CycleNET/example/training_dataset/singlecelltool_input_for_labeling_cellcycle.csv -p 1 -x 3 -y 4

Example (localization):
$ python src/generate_singlecell_coordinate_sheets.py -d example/labeled_images -o /home/username/git/CycleNET/example/training_dataset/singlecelltool_input_for_labeling_localization.csv -p 0 -x 3 -y 4
```

Script parameters:
```
  -d LABELED_DIRECTORY      Directory containing labeled images (segmentation output)
  -o OUTPUT_PATH            Output path (please use absolute path and include the filename)
  -p INDEX_IMAGE_PATH       Index position of the target image path in the *_labeled_coords.npy output file. The default is set to 0.
  -x INDEX_X                Index position of the center X-coordinate of the cell in the segmentation *_labeled_coords.npy output file. The default is set to 3.
  -y INDEX_Y                Index position of the center Y-coordinate of the cell in the segmentation *_labeled_coords.npy output file. The default is set to 4.
```

#### STEP 3 - LABEL CELLS
This is a custom-made GUI single cell labeling tool: https://github.com/BooneAndrewsLab/singlecelltool. 
Follow the instruction described in the repository page. The output generated from the previous step can be used as the 
"Cell data file" input in the tool. For the "Phenotype list" input, please use the following files accordingly:
<br/>
example/training_dataset/singlecelltool_input_labels_list_cellcycle.txt
<br/>
example/training_dataset/singlecelltool_input_labels_list_localization.txt

#### STEP 4 - GENERATE NEW TRAINING DATA
This script generates the train and test HDF5 files needed to train a new CycleNET network.
Please see example labeled cell files:
<br/>
example/training_dataset/labeled_cells_cellcycle.csv
<br/>
example/training_dataset/labeled_cells_localization.csv

```
Usage:
$ python src/generate_singlecell_coordinate_sheets.py -d <LABELED_DIRECTORY> 
  -i <INPUT_FILE> -t <TRAIN_FILE> -v <TEST_FILE> -x <INDEX_X> -y -<INDEX_Y>
  -c <LABELS_CELLCYCLE> -l <LABELS_LOCALIZATION> -s <CROP_SIZE> -n <CHANNEL> -m <METADATA> 
  -w <IMAGE_WIDTH> -z <IMAGE_HEIGHT> -r <SPLIT_RATIO>

Example (cell cycle):
$ python src/generate_new_training_data.py -d example/labeled_images 
  -i example/training_dataset/labeled_cells_cellcycle.csv -t /home/username/git/CycleNET/example/training_dataset/cellcycle_train_set.hdf5 
  -v /home/username/git/CycleNET/example/training_dataset/cellcycle_test_set.hdf5 -x 3 -y 4 
  -c -s 64 -n 5 -m 4 -w 1339 -z 1001 -r 0.8
  
Example (localization):
$ python src/generate_new_training_data.py -d example/labeled_images 
  -i example/training_dataset/labeled_cells_localization.csv -t /home/username/git/CycleNET/example/training_dataset/localization_train_set.hdf5 
  -v /home/username/git/CycleNET/example/training_dataset/localization_test_set.hdf5 -x 3 -y 4 
  -l -s 64 -n 5 -m 4 -w 1339 -z 1001 -r 0.8
```

Script parameters:
```
  -d LABELED_DIRECTORY      Directory containing labeled images (segmentation output)
  -i INPUT_FILE             File that contains the labeled cells. Required columns should be in the order of: Image Path, Center_X, Center_Y, Label. Use exported data from the single cell labeling tool (step 3).
  -t TRAIN_FILE             Output path for the training set (please use absolute path and include the filename)
  -v TEST_FILE              Output path for the test set (please use absolute path and include the filename)
  -p INDEX_IMAGE_PATH       Index position of the target image path in the *_labeled_coords.npy output file. The default is set to 0.
  -x INDEX_X                Index position of the center X-coordinate of the cell in the segmentation *_labeled_coords.npy output file. The default is set to 3.
  -y INDEX_Y                Index position of the center Y-coordinate of the cell in the segmentation *_labeled_coords.npy output file. The default is set to 4.
  -c LABELS_CELLCYCLE       Use this flag if the labels are cell cycle phases. Default is False.
  -l LABELS_LOCALIZATION    Use this flag if the labels are protein localization. Default is False.
  -s CROP_SIZE              Single cell crop size. Default is 64.
  -n CHANNEL                Number of channels/frames saved in the segmentation output *_labeled.npy file. Default is 5.
  -m METADATA               Number of metadata to include in the output file. Default is 4.
  -w IMAGE_WIDTH            Image width. Default is 1339.
  -z IMAGE_HEIGHT           Image height. Default is 1001.
  -r SPLIT_RATIO            Ratio to use when splitting the train and test datasets. Default is 0.8
```

After completing this step, the user can use the generated train and test datasets to train a new model. Kindly see 
instructions from "Training the Cell Cycle and Localization Networks" section.

---

### License
This software is licensed under the [BSD 3-Clause License][BSD3]. Please see the 
``LICENSE`` file for more details.

[cellcyclenet_models]: https://thecellvision.org/cellcycleomics/cyclenet_models.tar.gz
[cellcycle_training_dataset]: https://thecellvision.org/cellcycleomics/cellcycle_training_dataset.tar.gz
[localization_training_dataset]: https://thecellvision.org/cellcycleomics/localization_training_dataset.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause