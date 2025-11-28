JSON_FILES_DIR_NAME: str = "datasets" # Define the dataset JSON file name.
JSON_MODELS_DIR_NAME: str = "models" # Write the models file JSON name.

WINDOW_PERIODS: int = 14 # Define the market periods as a constant.
MINUTES_WINDOW: int = 15 # Constant of the elapsed time in each candle.
MARKET_SYMBOL: str = "TSLA" # Variable with the market symbol value.

# Define the Model Metadata (name, description, author).
NAME: str = f"{MARKET_SYMBOL} - Uncertainty Simple Perceptron"
DESCRIPTION: str = f"Uncertainty Simple Perceptron Model, for the market: {MARKET_SYMBOL} (risk 3:1 and {WINDOW_PERIODS} periods of {MINUTES_WINDOW} minutes)."
AUTHOR: str = "Dylan Sutton Chavez"

# Constant for the uncertainty (epsilon) of the model
EPSILON: float = 0.0007