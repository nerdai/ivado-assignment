# Overview
This repository houses the Python code that was used to conduct the ML modelling
analyses of the IVADO Labs take home assignment. The assignment itself was a
prototypical ML project requiring one to:
- perform EDA
- build an ML model
    - perform data cleaning
    - perform model selection
    - perform hyperparameter tuning
    - determine most appropriate performance metrics
- build a pipeline for reproducibility of results (or for deploying the model
to a server).

The rest of the README briefly summarizes those such activities taken here that
provide context to this code base. Before delving into that summary, however,
here is the print out of the tree-structure for this project.
```
.
├── Dockerfile
├── README.md
├── ¹artifacts
│   ├── models
│   │   ├── complete-model.joblib
│   │   └── imputed-model.joblib
│   ├── preds
│   │   ├── complete-predictions.csv
│   │   └── imputed-predictions.csv
│   └── training_logs
│       ├── complete.log
│       └── imputed.log
├── ¹data
│   ├── processed
│   │   ├── complete_df.csv
│   │   └── incomplete_df.csv
│   ├── raw
│   │   └── 2021-10-19_14-11-08_val_candidate_data.csv
│   └── splits
│       ├── complete_df
│       │   ├── test.csv
│       │   └── train.csv
│       └── incomplete_df
│           ├── test.csv
│           └── train.csv
├── ivado_assignment
│   ├── __init__.py
│   ├── bin
│   │   ├── __init__.py
│   │   ├── get_metrics.py
│   │   ├── inference.py
│   │   └── training.py
│   ├── data_processors
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── splitter.py
│   ├── settings
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── models.py
│   └── utils
│       ├── __init__.py
│       └── data_loader.py
├── notebooks
│   ├── sandbox-model.ipynb
│   └── sandbox.ipynb
├── poetry.lock
├── pylintrc
├── pyproject.toml
├── setup.sh
└── tests
    ├── __init__.py
    ├── test_data_cleaner.py
    ├── test_data_splitter.py
    └── test_ivado_assignment.py
```
¹ `data` and `artifacts` are not checked into this Github Repo.

## Exploratory Data Analysis
See `./notebooks/eda.ipynb` for the full analysis, but as a brief summary, 
listed below are the key takeaways:

1. There are a few data cleaning steps that need to be performed
2. We will conduct two separate analyses: `Complete` case and `Imputed` case
3. Target is slightly imbalanced with class N appearing only 31% of the time

`Complete` vs `Imputed`:
- Since there are rows with missing values, we decided to perform two separate
but similar analyses. 
    - `Complete` considers only complete observations
    - `Imputed` considers all observations and uses an impute strategy to fill
    in missing values

`laundry list` of data cleaning and processing items:
```
1. remove the two records that have erroneous entries for feature_6
2. drop feature 11 since same as feature_2
3. drop feature 12 since same as feature_4
4. cast feature_8 as a categorical feature
5. case feature_9 as a categorical feature
6. prepare a .csv file with only complete observations for complete-case analysis
7. prepare a .csv data file with both incomplete and complete observations for an imputed analysis
8. target has some imbalance with class N occurring only 31% of the time in the data
```

## Model Build
This section of README provides the details to the typical ML model building
process that was applied in this project. As noted above, some of the steps
were applied twice since two analyses were conducted: `Complete` and `Imputed`.

### Data Cleaning
The following processing steps were taken to the original data set file:

1. The two erroneously entered observations for `feature_6` were removed from
the dataset;
2. The `target` field was encoded to numeric with legend `Y=1` and `N=0`.

### Features
The features that were used in the analysis and which are organized by their
type are listed below:

```
numerical: ['feature_5', 'feature_6', 'feature_7'],

categorical: [
    'feature_0',
    'feature_1',
    'feature_2',
    'feature_3',
    'feature_4',
    'feature_8',
    'feature_9',
    'feature_10',
],
```

### Preprocessing
Preprocessing steps were applied uniformly to the type of features. I.e., all
`categorical` features used the same preprocessing steps.

`categorical`:
- Imputed with `missing` category (for `Imputed` analysis only);
- One-hot-encoder where first column is dropped.

`numerical`:
- Imputed with `median` value (for `Imputed` analysis only).

### Training
A (very) lightweight autoML was built and used for training an ML model in both
the `Imputed` and `Complete` case. Specifically, 3 candidate ML tree-based
models along with the `n_estimators` hyperparameter formed the search space
for our model selection.

#### Dataset Splitting
After running `ivado_assignment.data_processors.cleaner`, the `splitter` script
is ran in order to split the processed data files into `train` and `test`
splits. The default percentage used for both `Imputed` and `Complete` case was
75%/25% for `train`/`test`.

#### Choice of Metric
The Brier score is used here as the metric to select the best model as well as
for assessing the best model's performance on the `test` sets (although other
metrics are reported as well). The choice of Brier score was due to the fact
that the dataset is slight imbalanced, with class `0` or (`N`) being slightly
underepresented relative to its class counterpart, `1` or (`Y`). This metric
is understood as best practice for probability predictions even with
imbalanced data.

It should be noted though, that other metrics could be more suitable, depending
on the cost of false positives or false negatives on the minority class. The
package here can easily be adapted or slightly enhanced to accomodate such
metrics.

#### Models Tested and their HyperParameters
The following tree-based models and their hyperparameters were tested:

Models:
- `RandomForestClassifier` (`sklearn`)
- `GradientBoostingClassifier` (`sklearn`)
- `BalancedRandomForestClassifier` (`imblearn`) *uses undersampling*

HyperParameter:
- `n_estimator`: `[25, 50, 100]`

### Test Performance

`Complete`:
```
              precision    recall  f1-score   support

           0       0.63      0.51      0.57        37
           1       0.77      0.84      0.80        70

    accuracy                           0.73       107
   macro avg       0.70      0.68      0.68       107
weighted avg       0.72      0.73      0.72       107

Confusion:
 [[19 18]
 [11 59]] 

ROC-AUC:  0.7314671814671815
Log loss:  0.5551160202214863
Brier:  0.1878762585669782
F1 Score: 0.8027210884353742
```

`Imputed`:
```
              precision    recall  f1-score   support

           0       0.64      0.40      0.49        45
           1       0.77      0.90      0.83       100

    accuracy                           0.74       145
   macro avg       0.71      0.65      0.66       145
weighted avg       0.73      0.74      0.73       145

Confusion:
 [[18 27]
 [10 90]] 

ROC-AUC:  0.7006666666666667
Log loss:  0.5799618519385159
Brier:  0.18857150138888887
F1 Score: 0.8294930875576038
```

## Model Delivery (Docker)

For reproducing the model and the results, one can build the docker image
associated with the `Dockerfile` and the code herein. What follows below are
the steps required to build the model and produce predictions and get its
associated metrics against the respective `test` set. Note all commands are
expected to run while in the main folder of this repo and that it is assumed
that `docker-cli` has been installed.

#### Prerequisites
1. ensure raw data file is located in correct path

The raw data file `2021-10-19_14-11-08_val_candidate_data.csv`
must be stored in a directory: `ivado-assignment/data/raw/`.

#### Step 0: Clone the contents of this repository
`ssh`:
```
git clone git@github.com:nerdai/ivado-assignment.git
```

`https`:
```
git clone https://github.com/nerdai/ivado-assignment.git
```

#### Step 1: `cd` into the cloned repo and run `setup.sh`
```
cd ivado-assignment
sh setup.sh
```

#### Step 2: Build Docker Image
In this step, the code will be packaged into a docker image which we can then
use to:
- clean the data
- split it into train & test data sets
- train a model on the training set
- use the model to predict on the test set
- obtain the metrics of the model

*Note: we perform two analyses here: a complete-case analysis and one where
missing observations are imputed by a specified strategy.*

```
docker build -f Dockerfile . -t ivado-assignment
```

#### Step 3: Clean the data
Output of this step will be two `.csv` files, namely: `complete.csv` and
`incomplete.csv` that are stored in `./data/processed`.
```
docker run -v "$(pwd)/data/":/data ivado-assignment poetry run \
python -m ivado_assignment.data_processors.cleaner
```

#### Step 4: Split the data into train and test splits
Output of this step will be four `.csv` files, namely:
- `./data/splits/complete_df/train.csv`
- `./data/splits/complete_df/test.csv`
- `./data/splits/incomplete_df/train.csv`
- `./data/splits/incomplete_df/test.csv`

`Complete`:
```
docker run -v "$(pwd)/data/":/data ivado-assignment poetry run \
python -m ivado_assignment.data_processors.splitter \
--data ./data/processed/complete_df.csv --output ./data/splits/complete_df
```

`Imputed`:
```
docker run -v "$(pwd)/data/":/data ivado-assignment poetry run \
python -m ivado_assignment.data_processors.splitter \
--data ./data/processed/incomplete_df.csv --output ./data/splits/incomplete_df
```

#### Step 5: Training time
In this step, we run the commands to kickoff a lightweight autoML task that
will run through 3 different classifiers and perform a Bayesian Cross-Validation
to determine the number of estimators that should be used to achieve the best
value of model selection critiera. The best models are stored in
`./artifacts/models/`. Training logs are also stored in
`./artifacts/training_logs`.

`Complete`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.training \
--setting complete
```

`Imputed`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.training \
--setting imputed 
```

#### Step 6: Inferences (i.e., Make Predictions)
Here we load the previously trained models, and make predictions on the 
respective `test` sets. Predictions are stored in `./artifacts/preds/`.

`Complete`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.inference \
--data ./data/splits/complete_df/test.csv --setting complete
```

`Imputed`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.inference \
--data ./data/splits/incomplete_df/test.csv --setting imputed
```

#### Step 7: Get Performance Metrics
We pass the ground truth labels and the predictions to the `get_metrics`
job to produce a small report on the performance of the models on their test
sets.

`Complete`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.get_metrics \
--labels ./data/splits/complete_df/test.csv \
--preds ./artifacts/preds/complete-predictions.csv
```

`Imputed`:
```
docker run -v "$(pwd)/data/":/data -v "$(pwd)/artifacts/":/artifacts/ \
ivado-assignment poetry run python -m ivado_assignment.bin.get_metrics \
--labels ./data/splits/incomplete_df/test.csv \
--preds ./artifacts/preds/imputed-predictions.csv
```

## Model Delivery (Non-Docker, Poetry Only)
In this section, we provide similar steps (though we require one less step here
since we don't have to build a docker image) to the previous section for 
reproducing the analyses and model builds but without using Docker. 

#### Prerequisites
1. ensure correct version of Poetry is installed
2. ensure correct version of Python is installed and active
3. ensure raw data file is located in correct path

Since were are no longer using Docker, we have to ensure a few more dependencies
are installed and are of the correct versions. In particular, the steps below 
require:
- `poetry v. ^1.3.0` and
- `python v. ^3.11.0`
to be installed.

Installing or upgrading `poetry`:
```
# for linux, macOS
curl -sSL https://install.python-poetry.org | python3 -

# windows (powershell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Finally, the raw data file `2021-10-19_14-11-08_val_candidate_data.csv`
must be stored in a directory: `ivado-assignment/data/raw/`.

#### Step 0: Clone the contents of this repository
`ssh`:
```
git clone git@github.com:nerdai/ivado-assignment.git
```

`https`:
```
git clone https://github.com/nerdai/ivado-assignment.git
```

#### Step 1: `cd` into the cloned repo and run `setup.sh`
```
cd ivado-assignment
sh setup.sh
```

#### Step 2: Clean the data

```
poetry run python -m ivado_assignment.data_processors.cleaner
```

#### Step 3: Split the data into train and test splits

`Complete`:
```
poetry run python -m ivado_assignment.data_processors.splitter \
--data ./data/processed/complete_df.csv --output ./data/splits/complete_df  
```

`Imputed`:
```
poetry run python -m ivado_assignment.data_processors.splitter \
--data ./data/processed/incomplete_df.csv --output ./data/splits/incomplete_df 
```

#### Step 4: Training time
`Complete`:
```
poetry run python -m ivado_assignment.bin.training --setting complete
```

`Imputed`:
```
poetry run python -m ivado_assignment.bin.training --setting imputed
```

#### Step 5: Inferences (i.e., Make Predictions)

`Complete`:
```
poetry run python -m ivado_assignment.bin.inference \
--data ./data/splits/complete_df/test.csv --setting complete
```

`Imputed`:
```
poetry run python -m ivado_assignment.bin.inference \
--data ./data/splits/incomplete_df/test.csv --setting imputed
```

#### Step 6: Get Performance Metrics

`Complete`:
```
poetry run python -m ivado_assignment.bin.get_metrics \
--preds ./artifacts/preds/complete-predictions.csv \
--labels ./data/splits/complete_df/test.csv
```

`Imputed`:
```
poetry run python -m ivado_assignment.bin.get_metrics \
--preds ./artifacts/preds/imputed-predictions.csv \
--labels ./data/splits/incomplete_df/test.csv
```