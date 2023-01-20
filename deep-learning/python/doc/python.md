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
python3 -m pip install numpy matplotlib scikit-learn 
###progress bar
python3 -m pip install tqdm

#jupyter
python3 -m pip install jupyterlab

##pytorch
python3 -m pip install torch torchvision torchaudio torchviz

##cython
python3 -m pip install cython
python3 -m pip install pycocotools

## deactivate env
deactivate

##vscode
ctrl shift p
python select interpreter
=> select the virtual env interpreter

run in terminal => no frontend
run in windows => frontend ok

##pycharm
Not working well

activate python3 env
select interpreter /usr/bin/python3
interpreter stettings -> python interpreter
package: +
add all previous packages (except jupyterlab)

if problem, do: 
invalidate cache

Sometimes, we have the error:
This is normally a bug in some application using the D-Bus library.

  D-Bus not built with -rdynamic so unable to print a backtrace

  => redo the action
