# magical-movie-poster-processing

## Getting Started
Ensure you have Python 3.5 or greater installed. You can use pip or anaconda. You can download the latest version [here](https://www.python.org/downloads/).
#### 1. Clone the repository
Navigate to the folder in which you want to store this repository. Then clone the repository and change directory to the repository:
```
git clone https://github.com/jklewis99/magical-movie-poster-processing.git
cd magical-movie-poster-processing
```
#### 2. Activate a virtual environment (optional, but recommended):

##### With pip:
Windows
```
py -m venv [ENV_NAME]
.\[ENV_NAME]\Scripts\activate
```
Linux/Mac
```
python3 -m venv [ENV_NAME]
source [ENV_NAME]/bin/activate
```

##### With conda:
```
conda update conda
conda create -n [ENV_NAME]
conda activate [ENV_NAME]
conda install pip # install pip to allow easy requirements.txt install
```
#### 3. Install the requirements:
```
pip install -r requirements.txt
```

#### 3. Download missing image data
The image files are too large to store in this repo, so they must be downloaded externally. The JPEG image files can be found on [Kaggle](https://www.kaggle.com/raman77768/movie-classifier). Download this data (will default to `archive.zip`). Unzip this file into the designated `data\Images` folder in this repository:

* Linux/Mac
    ```
    unzip -qj /path/to/archive.zip "/Multi_Label_dataset/Images/*" -d /path/to/magical-movie-poster-processing/data/Images
    ```
    ###### Explanation of flags:
    * `q` suppresses printing of summary of each file extracted
    * `j` strips path info that is inside of the zipfile: (`Multi_Label_dataset/Images/*.jpg` becomes `/*.jpg`)
    * `d` destination directory for the the image files

#### 4. Download the pre-trained models
Models are too large to be stored on github, so each of them can be found on this [Google Drive folder](https://drive.google.com/drive/folders/10ism9rrmjRZlMwPBYz4FjyMGew32VVh4?usp=sharing). Save these files into the `/models` folder.

Alternatively, these files can be downloaded with the `gdown` executable (which is installed during the `pip install -r requirements.txt` command). The models can be downloaded with the following commands:

```
cd models
gdown https://drive.google.com/uc?id=1mflm_OPy-V1wVjG_zPqdcgjG0VfYDfV6
gdown https://drive.google.com/uc?id=1IAETAbjbFccrM56hQQYp8fL-sYQjpIJa
gdown https://drive.google.com/uc?id=1VYamYNnLz4mkvp9x4_3gttZqYQIZl4W3
```
**Note**: This is *required* for most CNN functions in `genre_classification.py` if you do not train your own models.

#### 5. (Optional) Download external raw data
The raw data, from Wei-Ta Chu and Hung-Jui Guo's paper **Movie Genre Classification based on Poster Images with Deep Neural Networks**, can be downloaded [here](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/Movie_Poster_Metadata.zip). For more information and to better understand the data, we recommend going to the [source](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html).

**Note**: For the file, [`poster_metadata.py`](/poster_metadata.py), to run, this data must be downloaded into the [`data`](/data) folder. It should be double nested folder, `Movie_Poster_Metadata/groundtruth`, in which are `.txt` files whose name is by their year of the films data within the `.txt` file, ranging from 1970 to 2015, inclusive. For example: `2002.txt` is a valid name of the file.

## Genre Classification
There are 4 modes when running `genre_classification.py`. `mode` is a REQUIRED parameter:

1. `train` will start loading data, creating neural network model, and training the model
2. `predict` will predict the genres based on the image
3. `find_threshold` will output and save the graph accuracy vs threshold under the name 'evaluation.png'
4. `class_activation_map` will create a class activation map on the image

There are 3 types of models when running `genre_classification.py`. 'NasNet' is the DEFAULT `model`:
1. `1` or `NasNet` will select NasNetLarge model
2. `2` or `InceptionResNet` will select InceptionResNetV2 model
3. `3` or `XceptionNet` will select XceptionNet model

When using `train`, there are 2 options for `train_mode` (default is 1)
1. `--train_mode=1` will train with the new model
2. `--train_mode=2` will train with the existing model

When using `predict` or `class_activation_map`, `--path`, representing the path of the image.

Use `python genre_classification.py --help` for all other parameters and options

Examples of Command:
* To create and train a new NasNetLarge model:

    ```
    python genre_classification.py train --model=1 --train_mode=1
    ```
* To use NasNetLarge model (assuming the model has been trained) to predict the image located at `data/Images/tt1077368.jpg`:
    ```
    python genre_classification.py predict --model=1 --path=data/Images/tt2911666.jpg
    ```
* To use NasNetLarge model (assuming the model has been trained) to create a class activation map on the image called `data/Images/tt2911666.jpg`:
    ```
    python genre_classification.py class_activation_map --model=1 --path=data/Images/tt2582802.jpg
    ```
* To use NasNetLarge model (assuming the model has been trained) to find the threshold (confidence level):
    ```
    python genre_classification.py find_threshold --model=1
    ```
### What to expect
If you are running these functions on a computer compatible with an NVIDIA GPU with enough memory, most functions will run relatively fast. In general, these functions will run on the CPU, but will be much slower:

For `train`, this is NOT recommended. *It is unreasonable to expect results within a week using only the CPU*

For `predict`, this will take about ~15-60 seconds, depending on the model

For `class_activation_map`, this will take approximately ~60 seconds.

For `find_threshold`, this is NOT recommended. *This will take approximately more than 30 minutes (for 1000 images).*

**Note**: Data size is downsampled if there is not enough available RAM for storing all of the images in the dataset.

## Box Office Prediction

There are 3 types of models when running `box_office.py`. `model` is a REQUIRED parameter:

1. `linear` will select Linear Regression model
2. `svr` will select Support Vector Regression model
3. `rfr` will select Random Forest Regression model

There are 4 types of kernels when running `box_office.py svr`. 'linear' is the DEFAULT `kernel`:
1. `linear` will select linear kernel
2. `poly` will select polynomial kernel
2. `rbf` will select rbf kernel
2. `sigmoid` will select sigmoid kernel

Use `box_office.py --help` for all other parameters and options

Examples of Command:
* To train, test, and show plots for predictions for a linear regression model:

    ```
    python box_office.py linear
    ```
* To train, test, and show plots for predictions for a Support Vector Regression model with a sigmoid kernel:

    ```
    python box_office.py svr --kernel=sigmoid
    ```

## Navigating this Repo
```bash
├── Genre Classification
│   ├── README
│   ├── genre_classification.py
│   └── requirements.txt
├── data
│   ├── Images
│   │   ├── *.jpg
│   ├── movies-metadata-cleaned.csv
│   ├── movies-metadata.csv
│   ├── posters-and-genres.csv
│   ├── test_data.csv
│   └── train_data.csv
├── figures
│   ├── *.png
├── models
│   ├── InceptionResNetV2.h5
│   ├── NasNetLarge.h5
│   ├── XceptionNet.h5
├── notebooks
│   ├── data_clean.ipynb
│   ├── data_overview.ipynb
│   └── pandas_intro.ipynb
├── utils
│   ├── background-img.png
│   ├── createbackground.py
│   ├── data_read.py
│   ├── misc.py
│   └── read_images.py
├── weights
│   ├── inception-resnet-v2.h5
│   └── xception_checkpoint-best.h5
├── box_office.py
├── clean.py
├── correlations_train_test.py
├── correlationsMetadata.py
├── generate_train_test.py
├── linearRegression.py
├── genre_classification.py
├── poster_metadata.py
├── requirements.txt
├── randomForests.py
├── README.md
├── svr.py
└── xception_transfer.py
```
> ### /data
>Inside of this folder is all of the data from each step in the process. After [raw metadata](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/Movie_Poster_Metadata.zip) is processed with the [`poster_metadata.py`](/poster_metadata.py) file, the file [`movies-metadata.csv`](/data/movies-metadata.csv) is created within this folder.

> ### /figures
>Inside this folder you will find figures that compare models and highlight features in the dataset. For example, you can see in [this image](/figures/background_img.png) that not all posters in the dataset came from movies. You can also see how predictions compare to actual labels in files that are identified by `NasNet-{}.png`

> ### /models
>Inside this folder you will find the `.h5` files that encode the pre-trained models for `InceptionResNetV2`, `NasNetLarge`, and `XceptionNet`. The weights/model file which is used in the `genre_classification.py` file is specified by the `model` parameter.

> ### /notebooks
>Inside this folder is the [notebook](/notebooks/pandas_intro) we created to help those new to the [Pandas](https://pandas.pydata.org/) data analysis API so that they can use this data for a *brief* introduction to the powerful tool. You will also find the [`data_clean.ipynb`](/notebooks/data_clean.ipynb) notebook, which was used to clean data for regression models.

> ### /utils
>This folder contains all utility modules and helper functions used across the repo.

> ### /weights
>This folder contains the weights for the XceptionNet model with different architecture. These weights are used by [`xception_transfer.py`](/xception_transfer.py).

All methods at the root of the repo were the primary functions used to create this repo.

## Acknowledgements
* Wei-Ta Chu and Hung-Jui Guo for the [raw data](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html)
* [raman](https://www.kaggle.com/raman77768) on Kaggle
* Dr. Eicholtz for the contiuous encouragement and commitment to his students' success.
