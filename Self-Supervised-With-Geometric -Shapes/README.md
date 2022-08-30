
### You have already done the following steps in JRDB_self_supervised_real_data Folder
##### Dependencies
##### Environmental setup
##### Activate Enviroment and Install a pakage
##### Download Data:
##### Data Preparation:


### Download Weights:
[Download_validation_lables](https://drive.google.com/drive/folders/1Y_lv_JI7xsaLDQBvSB7gZ0ThMZ7mrviF?usp=sharing) 

File name: SS3D_MOT_self_supervised_geometric_shapes.zip

### Update config file:
open config file, edit paths to root_dir, train_label_dir, test_label_dir, val_label_dir, load_weights, val_prediction_dir

```
config.yaml
```

### Perform evalulation:
you will get the results in val_prediction_dir, path defined in cofig file.

```
python -u eval_seq-ss.py

```

### Traning:
You can train with main.py
Note: it takes around 2 to 5 epochs to get the desires results. you will get the weights on every epoch.

```
python -u main.py

```



### Acknowledgement
Our code benefits from the implementations by [Zhang et al.](https://github.com/ZwwWayne/mmMOT) (Robust Multi-Modality Multi-Object Tracking) and [Sun et al.](https://github.com/shijieS/SST) (DAN-Deep Affinity Network)



