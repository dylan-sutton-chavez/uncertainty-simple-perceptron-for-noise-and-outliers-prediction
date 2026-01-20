# Uncertainty Simple Perceptron

> A linear perceptron leveraging Cover's Theorem via high-dimensional vectorization. It employs a modified step function with a defined epsilon threshold to quantify uncertainty, effectively filtering dataset noise. This document describes the architecture and usage in production.

## Key System Concepts

- **Uncertainty Step Function:** Three-state outputs (0.0, 0.5, 1.0) with configurable epsilon (ε) threshold: $ℎ(𝑥) = 0.0 (𝑥 < − ε), 0.5 (− ε ≤ 𝑥 ≤ ε), 1.0 (𝑥 > ε)$.

- **Early Stopping:** Built-in patience-based early stopping to prevent overfitting.

- **Object Based:** JSON and object based models saving, with metadata and timestamps.

- **Fine-tuning Capabilities:** Support for loading and continuing from saved models.

## System Architecture Background

**Core Components**

- **Cyclic Time:** Encode temporal values (minute, hour, day, month) using sin and cos transformations.

- **Indicators:** Compute technical indicators (Stochastic Oscillator, RSI, EMA).

- **Z-Score:** Normalize a given value, within a vector using the Z-Score standardization.

- **Pipeline:** Combine all feature encoders into a unified vector.

**Data Components**

- **Alpaca Client:** API client wrapper for fetching historical and real-time market data.

- **Persistence:** Lightweight JSON-inline lazy loading database for storing of data.

**Observability and Monitoring**

- **Better Stack:** An observability and monitoring platform, that combine graphs, logs, incidents management, and machine learning.

## Cloud Service Management

The system is deployed on Amazon Web Services (AWS), running on the Ubuntu operating system. You can use Remote SSH to manage your server in Visual Studio Code (VSC). For preference use an AWS in the region of us-east-1 and minimum a t3.small instance.

```bash
# Update the package list to ensure proper package management.
$ sudo apt update

# Install the latest available Python 3 version and verify the installation.
$ sudo apt install python3
$ python3 --version

# Install the pip packages manager and build-essential.
$ sudo apt install python3-pip build-essential

# Check if git is installed, and if not, install it.
$ sudo apt install git
```

## GitHub Private Repo Managment

```bash
# Clone the GitHub repository.
$ git clone https://github.com/dylan-sutton-chavez/aardvark-package.git
Username for 'https://github.com': your-github-user
Password for 'https://your-github-user@github.com': personal-access-token

# Move to the cloned GitHub repository and create the file for the datasets.
$ cd aardvark-package
$ mkdir datasets
```

> How to Obtain the Personal Access Token (PAC): GitHub Website → Settings → Developer Settings → Personal Access Tokens → Tokens (Classic) → Generate New Token.

## Libraries Installation (Python: 3.13.9+)

- numpy: 2.3.4
- alpaca-py: 0.43.2
- psutil: 7.1.3

```bash
# Install the libraries of the system.
$ pip install -r requirements.txt
```

## Environment Initialization

```bash
# Initialize the Alpaca client environment variables (API key and secret).
$ export ALPACA_KEY="a1b2c3d4e5f6g7h8i9j0"
$ export ALPACA_SECRET="a1b2c3d4e5f6g7h8i9j0"

# Initialize the Better Stack enviroment variables (Source → Logs → Python).
$ export BETTER_STACK_HOST="a1b2c3d4e5f6g7h8i9j0"
$ export BETTER_STACK_TOKEN="a1b2c3d4e5f6g7h8i9j0"
```

## Persistence System Launching

```bash
# Allows the process to keep running continuously even after the terminal session closes (nohup) — (run the program as a package '-m', without 'byte-compile' -B and desactivate the cache buffering '-u').
$ nohup python3 -u -B -m main
```

## System Reboot and Python Analysis Command

```bash
# Reboot the server (e.g., after applying system-level updates).
$ sudo reboot

# List all active Python 3 processes (useful for debugging or ensuring the system is not running multiple instances).
$ ps aux | grep "python3"

# Terminate all active Python 3 processes.
$ sudo pkill -f python3

# Remove the local repository ONLY if you intentionally want a fresh clone (delete all contents → -rf).
$ rm -rf aardvark-package

# Shows all the content of a file and print in the output.
$ cat <file-name>

# Read the last ten lines of a file and print in the output. It runs in real time, monitoring changes and printing them in the output.
$ tail -f <file-name>
```

## Proprietary Agreement License

Permission is hereby granted, free of charge, to use, copy, and distribute software.

## Backtest

The system has been tested in: backtest.pdf. Some libraries/packages have been renamed or refactored within the repository, but the math is the same.
