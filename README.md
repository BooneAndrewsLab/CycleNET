# CellCycleNET



### Project Setup

Clone the repository
```
$ git clone https://github.com/BooneAndrewsLab/CellCycleNET.git
$ cd CellCycleNET
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

Create a virtual environment and install requirements
```
$ virtualenv --python=python2.7 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

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

### Prerequisites
Python 2.7 https://www.python.org/downloads

### License
This software is licensed under the [BSD 3-Clause License][BSD3]. Please see the 
``LICENSE`` file for more details.

[cellcyclenet_models]: https://thecellvision.org/cellcycleomics/cyclenet_models.tar.gz
[cellcycle_training_dataset]: https://thecellvision.org/cellcycleomics/cellcycle_training_dataset.tar.gz
[localization_training_dataset]: https://thecellvision.org/cellcycleomics/localization_training_dataset.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause