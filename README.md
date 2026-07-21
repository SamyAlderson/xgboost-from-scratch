# xgboost-from-scratch
A lightweight Python implementation of the XGBoost algorithm from scratch

## What it does

This project provides a basic implementation of the XGBoost algorithm in Python. It includes a decision tree implementation, custom metric calculation, and gradient calculation and optimization. The goal is to create a simple and efficient XGBoost algorithm without relying on external libraries.

## Installation

To install the project, run the following command:
```bash
pip install .
```
This will install the necessary dependencies and make the package available for import.

## Usage

To use the XGBoost algorithm, simply import the package and call the `train` function:
```python
from xgboost_from_scratch import XGBoost

# Create an XGBoost instance
model = XGBoost()

# Train the model on some data
model.train(X, y)
```
Replace `X` and `y` with your actual dataset and target variable.

## Building from source

To build the project from source, clone the repository and install the required dependencies:
```bash
git clone https://github.com/SamyAlderson/xgboost-from-scratch.git
cd xgboost-from-scratch
pip install -r requirements.txt
```
## Running tests

To run the tests, execute the following command:
```bash
pytest
```
This will run the test suite and report any failures or errors.

## Project structure

Here's an overview of the key files in the project:

* `xgboost.py`: The main implementation of the XGBoost algorithm
* `tree.py`: A decision tree implementation used by the XGBoost algorithm
* `metric.py`: Custom metric calculation for the XGBoost algorithm
* `gradient.py`: Gradient calculation and optimization for the XGBoost algorithm
* `tests`: The test suite for the project

## License

Copyright (c) 2026 SamyAlderson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.