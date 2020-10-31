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
```
#### 3. Install the requirements:
```
pip install -r requirements.txt
```

#### 3. Download missing image data
The image files are too large to store in this repo, so they must be downloaded externally. The JPEG image files can be found on [Kaggle](https://www.kaggle.com/raman77768/movie-classifier). Download this data (will default to `archive.zip`). Unzip this file into the designated `data\Images` folder in this repository:
#### Linux/Mac
```
unzip -qj /path/to/archive.zip "/Multi_Label_dataset/Images/*" -d /path/to/magical-movie-poster-processing/data/Images
```
###### Explanation of flags:
* `q` suppresses printing of summary of each file extracted
* `j` strips path info that is inside of the zipfile: (`Multi_Label_dataset/Images/*.jpg` becomes `/*.jpg`)
* `d` destination directory for the the image files

## Branches
After you have cloned the repository and have set up the data, checkout to your branch (`lewis`, `kim`, `baehr`, `phan`):
```
git checkout [BRANCH-NAME]
```
Show all the branches and verify that you are on the right branch:
```
git branch -a
```
Before you begin working on any work, make sure your branch is in sync with `main`:
```
git checkout main
git pull
git checkout [BRANCH-NAME]
git merge main
```
When you want to merge your branch with main:
```
git add . # stage all files for commit
git commit -m "Statement describing commit"
```
* If this is your first push to a remote upstream for the branch:
    ```
    git push --set-upstream origin [BRANCH-NAME]
    ```
* Otherwise:
    ```
    git push
    ```
```
git checkout main
git merge [BRANCH-NAME]
git commit -m "Merged branch [BRANCH-NAME] with main"
git push
```
### Acknowledgements
* Wei-Ta Chu and Hung-Jui Guo for the [raw data](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html)
* [raman](https://www.kaggle.com/raman77768) on Kaggle