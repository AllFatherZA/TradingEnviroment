Trading Environment Simulation
This project simulates a trading environment using historical market data. It includes a custom environment for trading, a reinforcement learning setup, and tools for visualizing trading performance. The environment allows users to test trading strategies, simulate trades, and analyze results.

Table of Contents
Features

Installation

Usage

Code Structure

Dependencies

Contributing

License

Features
Custom Trading Environment: Simulates a trading environment with historical market data.

Trading Strategies: Supports buy/sell actions with stop-loss and drawdown management.

Visualization: Provides real-time visualization of trading performance.

Machine Learning Integration: Uses polynomial regression for price forecasting.

Performance Metrics: Tracks accuracy, net worth, profit/loss, and other key metrics.

Installation
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/trading-environment.git
cd trading-environment
Install Dependencies:
Ensure you have Python 3.8+ installed. Then, install the required packages:

bash
Copy
pip install -r requirements.txt
Download Data:
Place your historical market data (e.g., US30(10).csv) in the project directory.

Usage
Running the Simulation
To run the trading simulation, execute the following command:

bash
Copy
python main.py
Customizing the Environment
You can customize the trading environment by modifying the parameters in the CustomEnv class:

python
Copy
env = CustomEnv(
    df=df, 
    crystalball=1000, 
    drawdown_p=0.95, 
    df_normalized=df, 
    lot_size=0.02, 
    account_size=1, 
    decimals=1, 
    spread=6, 
    multiple=1, 
    conversion=17.99, 
    stoploss=60
)
Visualizing Results
The simulation includes a visualization tool to display trading performance in real-time. Enable visualization by setting visualize=True in the random_games function:

python
Copy
random_games(env, visualize=True)
Code Structure
CustomEnv Class: The core trading environment. Handles trading logic, state management, and performance tracking.

random_games Function: Simulates random trading episodes for testing and evaluation.

get_action Function: Implements a simple trading strategy using polynomial regression for price forecasting.

TradingGraph Class: Provides visualization of trading performance (imported from utils).

Dependencies
Python 3.8+

Libraries:

pandas

numpy

matplotlib

scikit-learn

scipy

deque (from collections)

Install all dependencies using:

bash
Copy
pip install pandas numpy matplotlib scikit-learn scipy
Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Submit a pull request with a detailed description of your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.