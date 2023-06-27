# CellCycleNet



### Project Setup

Clone the repository
```
$ git clone https://github.com/BooneAndrewsLab/CellCycleNet.git
$ cd CellCycleNet
```

Create a virtual environment and install requirements
```
$ virtualenv --python=python2.7 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Download: Cell Cycle and Localization networks: [cellcyclenet_models.tar.gz][cellcyclenet_models.tar.gz]

### Pipeline: Single Cell Segmentation + Cell Cycle and Localization Prediction

#### Single Cell Segmentation
```
$ python src/segmentation.py -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -s <SCRIPTS_FOLDER> -g <GFP_CHANNEL> -n <NUCLEAR_CHANNEL> -c <CYTO_CHANNEL>
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

#### Cell Cycle and Localization Prediction
```
$ python resnet_evaluate_whole_screen.py -l <LOC_CPKT> -c <CYC_CPKT> -s <SCREENSTOANALYZE> -o <OUTPATH> -n
```

Script parameters:
```
  -l LOC_CPKT           Path to model/checkpoint for localization network
  -c CYC_CPKT           Path to model/checkpoint for cell cycle network
  -s SCREENSTOANALYZE   Screen/s to analyze
  -o OUTPATH            Where to store output csv files
  -n                    Boolean: store True if specified. Use non cropped cells
```
_This script calls src/**preprocess_images**.py and src/**input_queue_whole_screen**.py_

### Prerequisites
Python 2.7 https://www.python.org/downloads

### License
This software is licensed under the [BSD 3-Clause License][BSD3]. Please see the 
``LICENSE`` file for more details.

[cellcyclenet_models.tar.gz]: https://thecellvision.org/cellcycleomics/network_models/cellcyclenet_models.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause