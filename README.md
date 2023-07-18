# CellCycleNET



### Project Setup

Clone the repository
```
$ git clone https://github.com/BooneAndrewsLab/CellCycleNET.git
$ cd CellCycleNET
```

Create a virtual environment and install requirements
```
$ virtualenv --python=python2.7 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Pipeline: Single Cell Segmentation + Cell Cycle and Localization Prediction

#### Single Cell Segmentation
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

#### Compile single cell crops and coordinates
```
Usage: 
$ python src/compile_single_cells.py -l <LABELED_FOLDER> -i <IMAGE_FOLDER> -s <CROP_SIZE> -g <GFP_CHANNEL> -n <NUCLEAR_CHANNEL> -c <CYTO_CHANNEL>

Example:
$ python src/compile_single_cells.py -l example/labeled_images -i example/input_images -s 64 -g ch1 -n ch2 -c ch3
```

#### Cell Cycle and Localization Prediction

Download and save [Cell Cycle and Localization networks][cellcyclenet_models.tar.gz] to 'models' folder
```
$ mkdir models
$ wget -P models https://thecellvision.org/cellcycleomics/network_models/cellcyclenet_models.tar.gz
$ tar -xvzf models/cellcyclenet_models.tar.gz --directory=models
```

Run evaluation and prediction script 
```
Usage:
$ python src/resnet_evaluate_whole_screen.py -l <LOC_CPKT> -c <CYC_CPKT> -i <INPUT_PATH> -o <OUTPATH> -n

Example:
$ python src/resnet_evaluate_whole_screen.py -l models/localization/inference_leo_model.ckpt-6500 -c models/cellcycle/inference_oren_model.ckpt-9500 -i example/labeled_images -o example/predictions
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

[cellcyclenet_models.tar.gz]: https://thecellvision.org/cellcycleomics/network_models/cellcyclenet_models.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause