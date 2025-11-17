# Aardvark — Uncertainty Simple Perceptron

> A linear perceptron leveraging Cover's Theorem via high-dimensional vectorization. It employs a modified step function with a defined epsilon threshold to quantify uncertainty, effectively filtering dataset noise.

## Key Concepts

- **Uncertainty Step Function:** Three-state output (0, 0.5, 1) with configurable epsilon threshold.

- **Early Stopping:** Built-in patience-based early stopping to prevent overfitting.

- **Model Persistence:** JSON-based model saving with metadata and timestamps.

- **Fine-tuning Capability:** Support for loading and continuing from saved models.

## Architecture Background

**Core Components:**

- **Uncertainty Simple Perceptron:** Main perceptron implementation with training, inference, and model managment.

**Vectorization:**

- **Cyclic Time Encoder:** Encode temporal values (minute, hour, day, month) using sin/cos transformations.

- **Time Series:** Computes technical indicators (Stochastic Oscillator, RSI, EMA).

- **Z-score:** Normalize a given value, whit a vector using the Z-score standarization.

- **Features Vectorizer:** Combine all feature encoders into a unifed vector.

**Data Components**

- **Alpaca Markets:** API client for fetching historical and real-time market data.

- **DuckDB:** Lightweight JSON-based database for storing datasets and configurations.

## Installation

- Python==3.13.9
  - numpy==2.3.4
  - alpaca-py==0.43.2
  - dotenv==0.9.9

**Installation**

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

## Configuation

- **Time Series Configuration:** Raw vector and periods for technical indicators.

- **Time Configuration:** Current time components for cyclic encoding.

- **Normalization Configuration:** Z-score object and values for normalization.

## Usage

Referencing "aardvark.ipynb", where implement the complete pipeline of: vectorization, training, test,...

Show the complete pipeline:

1. Data fetching from Alpaca Markets.

2. Dataset splitting (training/test).

3. Z-score calculation and configuration.

4. Feature vectorization with stop-loss/take-profit labels.

5. Model training with early stopping.

6. Inference and evaluation.

## Model Output

- **Output = 1:** Positive.

- **Output = 0.5:** Uncertainty region.

- **Output = 0:** Negative.

## Technical Details

**Algorithm:** The algorithm is based in the simple perceptron and Cover's theorem.

**Time Complexity:**

- **Training:** O(e * n * d)

- **Inferenece:** O(d)

## License

The software is licensed under a proprietary agreement, maintaining closed source distribution
