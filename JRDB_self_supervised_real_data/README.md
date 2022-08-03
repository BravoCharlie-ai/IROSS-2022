
# Self-supervised_Real_Dataset

### Dependencies
cuda_10.2 <br/>
cudnn_v7.6.5

### Environmental setup


```
conda env create -f environment.yml
```

### Activate Enviroment and Install a pakage

```
conda activate pcdan_iros2022
```


### Download Data:

[Download](https://jrdb.stanford.edu/dataset/about)

### Data Preparation:
edit prepare_dataset.py
run prepare_dataset.py after giving it path to the downloaded train and test dataset. Also give path where you want to save the processed data
Note: you will get following folders Sequences and Test_sequences which will contained lidar frames only.

```
python -u prepare_dataset.py
```

### Download Weights:

[Download_Weights](https://drive.google.com/file/d/1A52WxMKAxnHaVpCAvhXoNHonq8IZcCRQ/view?usp=sharing)


### Download Validation Lables:

[Download_validation_lables](https://drive.google.com/drive/folders/1Y_lv_JI7xsaLDQBvSB7gZ0ThMZ7mrviF?usp=sharing) 

### Update config file:
open config file, edit paths to root_dir, train_label_dir, test_label_dir, val_label_dir, load_weights, val_prediction_dir

root_dir takes the path to processed dataset (Train and Test data) genereted after running prepare_dataset.py
train_label_dir takes the path to labels in the downloded train dataset (JSON files). Path would be something like, 'train_dataset_with_activity/labels/labels_3d/'
test_label_dir takes the path to detections of downloaded test dataset(JSON files).Path would be something like, 'test_dataset_without_labels/detections/detections_3d/'
val_label_dir takes the path to downloaded validation labels from google drive. These are txt files used to compare validtion prediction with these ground truths.
load_weights takes to path to downloaded weights from google_drive. Path can be something like 'SS3D_MOT_self_supervised_weights.zip'
val_prediction_dir takes the path where you want to save your validation results 

```
config.yaml
```

### Perform evalulation:
you will get the results in val_prediction_dir, path defined in cofig file.

```
python -u eval_seq.py

```

### Traning:
You can train with main.py
Note: it takes around 10 to 15 epochs to get the desires results. you will get the weights on every epoch.

```
python -u main.py

```



### Acknowledgement
Our code benefits from the implementations by [Zhang et al.](https://github.com/ZwwWayne/mmMOT) (Robust Multi-Modality Multi-Object Tracking) and [Sun et al.](https://github.com/shijieS/SST) (DAN-Deep Affinity Network)


