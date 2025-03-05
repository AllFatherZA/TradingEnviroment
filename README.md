# README.md

## Custom Trading Environment

This repository contains a custom trading environment written in Python for creating and testing trading strategies. The environment is designed to simulate trading scenarios and evaluate the performance of trading algorithms.

### Introduction

The custom trading environment is built using the `CustomEnv` class and includes various functionalities for backtesting and evaluating trading strategies. The environment allows users to interact with historical price data, define trading parameters, and simulate trades based on predefined actions.

### Requirements

To run the custom trading environment, you will need the following:

- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `finta`, `statsmodels`, `sklearn`

### Getting Started

1. Clone this repository to your local machine.

2. Ensure you have Python 3.x installed.

3. Install the required Python libraries using the following command:

```
pip install pandas numpy matplotlib scipy finta statsmodels scikit-learn
```

### How to Use

1. Import the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy import signal
from  scipy.optimize import OptimizeWarning
import warnings
from finta import TA
import os
from utils import TradingGraph, Write_to_file
from datetime import date, datetime, timedelta
import math
from collections import deque
import neat
import pickle as pickle
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import random 
from sklearn.preprocessing import PolynomialFeatures
```

2. Create an instance of the `CustomEnv` class:

```python
df = pd.read_csv('../US30(10).csv')  # Load historical price data
df.columns=['Date','Open','High','Low','Close','Volume']  # Rename columns if necessary
df['CloseDiff'] = df['Close'].diff()  # Calculate price differences
df = df[3000:]  # Adjust data range as needed

# Initialize the trading environment
env = CustomEnv(df=df, Crystalball=1000, Drawdown_P=0.95, df_normalized=df, LotSize=0.02, AccountSize=1, Decimals=1, Spread=6, Multiple=1, Convertion=17.99, Stoploss=60)
```

3. Use the trading environment to run simulations and test trading strategies:

```python
Days = 0

# Run random games for testing
Random_games(env, visualize=True)
```

### Important Notes

- The custom trading environment is meant for educational and experimental purposes only. It does not constitute financial advice or a guarantee of profitable trading strategies.

- The environment includes sample actions (`getaction()`) for buying, selling, or holding positions based on price forecasts. You may replace this function with your custom trading strategy.

- Ensure that you have the necessary historical price data in the format expected by the environment (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`). Adjust the data range and columns as needed to match your dataset.

### License

This custom trading environment is licensed under the [MIT License](LICENSE).
