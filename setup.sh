#!/bin/bash
VENV_DIR=${1:-".venv/mmagic"}
conda env create -f conda.yaml --prefix $VENV_DIR
conda run --no-capture-output -p $VENV_DIR mim install -r requirements.txt -e .
conda run --no-capture-output -p $VENV_DIR mim install albumentations
