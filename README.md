# Sampling Demon

Official implementation of Sampling Demon, arxiv:2410.05760

## Installation
Please run the following commands to install the required packages.
```
conda env create -f environment.yml # The build takes 30 minutes on our machine :(
pip install image-reward
pip install -e .
```
If torch versioning issue occur, remove torch-related packages and reinstall `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1` is a usual practice.
If it does not work, please try to install the packages in the `environment.yml` one by one. 


The `environment.yml` is written by the command
```
conda env export > environment.yml --no-builds 
```

## Usage