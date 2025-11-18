# Aardvark — Uncertainty Simple Perceptron

> This document describes the architecture and usage of “Aardvark” in production, referring to the files (monitor.py, vectorize_and_train.py).

## Production Architecture

- Training and Vectorization (vectorize_and_train.py): Handles training, including functions for ingesting raw data, performing vectorization, applying labeling, and training (or retraining) a model object.

- The System (monitor.py): This module is responsible for ingesting data within 14-candle windows (each candle is 15 minutes) and running predictions with the model, placing trades in the market. Finally, it retrains the model every "n" days to adapt to the market.

## Installation

- Python==3.13.9
  - numpy==2.3.4
  - alpaca-py==0.43.2
  - dotenv==0.9.9

```bash
# Clone the Repository.
git clone https://github.com/dylan-sutton-chavez/aardvark-uncertainty_simple_perceptron
cd aardvark-uncertainty_simple_perceptron

# Instal dependecies.
pip install -r requirements.txt
```

**Set Enviroment Variables**

Create a file named ".env", and set the next enviroment variables.

```bash
ALPACA_KEY=1234567890
ALPACA_SECRET=1234567890
```

## License

The software is licensed under a proprietary agreement, maintaining closed source distribution
