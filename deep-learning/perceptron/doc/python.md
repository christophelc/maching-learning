# Prepare virtual env

## install libraries for ubuntu
sudo apt install python3-venv

## prepare env
python3 -m venv python/env

## update gitignore
cd python
vi .gitignore
env

#use python virutal env

## actvate env
source env/bin/activate

## install ml lib in virtual env
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install scikit-learn


## deactivate env
deactivate

