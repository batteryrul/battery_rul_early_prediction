# Battery Cycle Life Prediction

This project is for paper "A Hybrid Ensemble Deep Learning Approach for Early Prediction of Battery Remaining Useful Life"

## Getting Started

The original data is available at https://data.matr.io/1/, download the data and put them into folder /Data

### Prerequisites

1.After downloading the data, run  BuildPkl_Batch1.py, BuildPkl_Batch2.py and BuildPkl_Batch3.py to extract the data for training and test

```
python BuildPkl_Batch1.py
```

2.Run Load_Data.py to delete bad battery data.

```
python Load_Data.py
```

Note: BuildPkl_Batch*.py and Load_Data.py are provided by author, small changes are made. 
Original code: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

### Feature Extraction

feature_selection.py provide the implementation of feature extraction and selection


## Test with different models using feature_based model

feature_based_model_paper.py includes several model implementation with different features combination.
--feature: variance, discharge, full
--model: elastic,SVR,RFR,AdaBoost,XGBoost

```
python feature_based_model_paper.py --feature=0 --model=0
```

### Proposed hybrid model with snaphsotensemble

pytorch_hybrid_model_snapshot_train.py includes hybrid model implementation. 
Before running, need to generate the dataset via data_process_for_hybrid_model.py.

```
python pytorch_hybrid_model_snapshot_train.py
```
