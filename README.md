# xgboost-from-scratch
A Python implementation of the XGBoost algorithm from scratch

## What/Why
This project is a self-contained implementation of the XGBoost algorithm in Python. The goal is to provide a clear and concise understanding of how XGBoost works, without relying on external libraries or frameworks.

## Install
To use this project, you'll need to have Python 3.8 or later installed on your system. You can install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
## Usage
To train an XGBoost model, simply run the following command:
```bash
python src/main.py
```
This will train a model on the `data/train.csv` file and save the results to `data/model.json`.

To make predictions using the trained model, run the following command:
```bash
python src/main.py predict
```
This will load the trained model from `data/model.json` and make predictions on the `data/test.csv` file.

## Build from Source
To build the project from source, simply run the `setup.py` script:
```bash
python setup.py build
```
This will compile the project and place the resulting binaries in the `dist/` directory.

## Project Structure
The project is organized as follows:

* `src/`: contains the implementation of the XGBoost algorithm
* `tests/`: contains unit tests for the implementation
* `data/`: contains sample data for training and testing
* `requirements.txt`: contains the list of dependencies required by the project

## License
This project is released under the MIT License.

## Features

* Gradient boosting
* Decision tree implementation
* XGBoost algorithm from scratch
* Custom metric calculation
* Gradient calculation and optimization

## Dependencies

* numpy
* scipy
* pandas