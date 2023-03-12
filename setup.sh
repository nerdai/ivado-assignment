#!/bin/sh

# make dir for storing preds and models
mkdir artifacts 
mkdir artifacts/preds
mkdir artifacts/models
mkdir artifacts/training_logs

# make dirs for data
mkdir data
mkdir data/processed
mkdir data/splits
mkdir data/splits/complete_df
mkdir data/splits/incomplete_df