# Applications of Machine Learning in Earthquake Prediction
## Overview
### Goal
The objective of this project is to explore the potential of machine learning techniques in analyzing and predicting key earthquake parameters based on observed or estimated data. Earthquake prediction is a highly complex and nonlinear problem, where traditional physics-based models often fall short in flexibility and adaptability. Machine learning offers a data-driven alternative capable of capturing hidden patterns and complex relationships among seismic variables.

### Subtasks
* Predict magnitude based on intensity map, epicenter location and depth.
* Predict depth based on intensity map, epicenter location and magnitude.
* Predict epicenter location based on intensity map.
* Predict intensity distribution of an earthquake event bsed on magnitude, distance, site condition, and style-of-faulting (斷層分類).

> The first three tasks are located in the `Predict (Regression)` folder, and the last task is located in the `Intensity Generator` folder.

### Data sources
[地震活動彙整 - 中央氣象署地震測報中心](https://scweb.cwa.gov.tw/zh-tw/earthquake/data)

## Prerequisite
* Python 3.13.3
* Packages: in [requirements.txt](https://github.com/douobb/2025-AI-FinalProject/blob/main/requirements.txt)

## Usage
### Download & Preprocess
Execute the following files in order:
1. `image_download.py` - Download images from the web  
2. `preprocess.py` - Preprocess and crop the images  
3. `data folder.py` - Split images into train/test folders  

After execution, the following folder structure should be created:
```
data/
├── raw_images/
│   ├── 2000/
│   │   ├── 2000001.png
│   │   ├── 2000002.png
│   │   └── ...
│   ├── ...
│   └── 2025/
├── processed_images/
│   ├── 2000/
│   ├── ...
│   └── 2025/
├── processed_images_no_epicenter/
│   ├── 2000/
│   ├── ...
│   └── 2025/
```

```
data_split/
├── processed_images/
│   ├── train/
│   │   ├── 2000001.png
│   │   ├── 2012015.png
│   │   └── ...
│   └── test/
│       ├── 2004012.png
│       ├── 2023007.png
│       └── ...
├── processed_images_no_epicenter/
│   ├── train/
│   └── test/
```
### For Predict (Regression)
1. Before starting, make sure that the `data_split` folder, `Earthquake.csv`, and the following Python files are in the same directory: `utils.py`, `CNN.py`, and `main.py`.
2. In `main.py`, you can configure the following hyperparameters:
   | Parameter | Possible Value | Description |
   | --- | --- | --- |
   | `target` | `'dep'`, `'mag'`, `'loc'`, `'lat'`, `'lon'` | The prediction target (e.g., depth, magnitude, or location) |
   | `nRound` | Positive integer | Number of repeated training rounds for averaging results |
   | `EPOCHS` | Positive integer | Number of training epochs per round |
   | `lr` | Small float | Learning rate used by the optimizer |
   | `model_type` | `'resnet18'`, `'efficientnet_b0'`, `'simple_cnn'` | The model architecture used for processing the input images |
   | `use_extra_params`| `True`, `False` | Whether to include additional input features (e.g., magnitude, depth)|
   | `criterion` | `nn.L1Loss()`, `nn.SmoothL1Loss()`, `nn.MSELoss()` | Loss function used during training |
   
4. Run `main.py`. For each round, a `.csv` file will be generated that records both the predicted and actual values on the test set.
5. After all rounds are completed, the script will print the average MAE across all rounds.


### For Intensity Generater
Execute the following tasks in order:
1. copy or moved the `data_split/processed_no_epiceter` folders to current folder  
2. create `sea_removed` and `prediction` folder and their subfolders following the provided structure
3. execute `remove_sea_color.py`. You should see modified images appearing in the `sea_removed`

Ensure the following structure:
```
data_split/
├── sea_removed/
│   ├── train/
│   │   ├── 2000001.png
│   │   ├── 2012015.png
│   │   └── ...
│   └── test/
│       ├── 2004012.png
│       ├── 2023007.png
│       └── ...
├── processed_images_no_epicenter/
│   ├── train/
│   └── test/
|
├── prediction/
│   ├── train/
│   └── test/
```
There are two options: to run locally, or using cloud services.

1. Run `main_test.py`, here everything is packed together. Use this if you are using cloud services.
   Note: In Google Colab, you must mount with Google drive, and upload `main_test.py`,, as well as the `data_split/sea_removed/` and `data_split/prediction/` folder in the running directory.
2. Run "main_test.py" locally.

After all rounds are completed, you should see the production of `loss.png` image, and `IntensityNet_model.pth` storing trained model parameters. 

Furthermore, you should be able to inspect the predictions the model has made as images in the `data_split/prediction/` folder.


