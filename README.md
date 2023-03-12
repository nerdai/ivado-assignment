# Overview

## Exploratory Data Analysis

## Model Build

### Data Cleaning

### Features

### Training
#### Dataset Splitting

### Performance

#### Choice of Metric

## Model Delivery (Docker)

For reproducing the model and the results, one can build the docker image
associated with the `Dockerfile` and the code herein. What follows below are
the steps required to build the model and produce predictions and get its
associated metrics against the respective `test` set. Note all commands are
expected to run while in the main folder of this repo and that it is assumed
that `docker-cli` has been installed.

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