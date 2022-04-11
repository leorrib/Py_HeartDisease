# Machine Learning - predicting the existence of a Heart Disease in a patient
This project returns 2 machine learning models that predict, given some data, if a patient has heart disease. One of the models use Support Vector Machine, while the other one makes use of Random Forest Classifier.

## Visualizing the result
Just open the main.ipynb file on github itself and you will be able to see the script and some metrics of the Machine Learning models created.

## How to run

### Requirements
There are two ways to generate the ml models: the first one is by using the jupyter notebook file (main.ipynb), while the second only requires a Python installation.

- Python 3
- Jupyter Notebook

### Step-by-step - common steps
- Clone the project
- On the root dir, create virtual environment (python3 -m venv .venv)
- On the same dir, start the virtual environment (source .venv/bin/activate)
- Install packages listed on requirements.txt (pip3 install -r requirements.txt)
- The variables used by the scripts are on the parameters.json

#### Step-by-step - jupyter notebook file
- Create a jupyter kernel (ipython kernel install --user --name=.venv)
- Start jupyter notebook, open the main.ipynb file, select the .venv kernel and then run all the cels.
- Make sure you delete the kernel once you are done (jupyter-kernelspec uninstall .venv).

#### Step-by-step - python file
- On the root dir, enter python3 'main.py'.