"""
This file (monitor.py) is the main orchestrator of the 'Aardvark' system. 
It handles two operating cycles: real-time inference and periodic retraining.

In its inference cycle, the monitor fetches market data (the 14-candle window) every fifteen minutes. 
It loads the serialized model, vectorizes the window data, and executes a prediction.

Every 'n' days, the monitor runs the retraining module (vectorize_and_train.py). 
This process ingests new historical data and adjusts the model to current market conditions.
"""