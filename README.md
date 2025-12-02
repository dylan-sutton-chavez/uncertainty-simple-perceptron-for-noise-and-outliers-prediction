# Aardvark Package — The Uncertainty Simple Perceptron System

> A linear perceptron leveraging Cover's Theorem via high-dimensional vectorization. It employs a modified step function with a defined epsilon threshold to quantify uncertainty, effectively filtering dataset noise. This document describes the architecture and usage of "Aardvark" in production.

## Key System Concepts

- **Uncertainty Step Function:** Three-state outputs (0.0, 0.5, 1.0) with configurable epsilon (ε) threshold: $ℎ(𝑥) = 0.0 (𝑥 < − ε), 0. 5 (− ε ≤ 𝑥 ≤ ε), 1.0 (𝑥 > ε)$.

- **Early Stopping:** Built-in patience-based early stopping to prevent overfitting.

- **Object Based:** JSON and object based models saving, with metadata and timestamps.

- **Fine-tuning Capabilities:** Support for loading and continuing from saved models.

## System Architecture Background

**Core Components**

- **Cyclic Time:** Encode temporal values (minute, hour, day, month) using sin and cos transformations.

- **Indicators:** Compute technical indicators (Stochastic Oscillator, RSI, EMA).

- **Z-Score:** Normalize a given value, within a vector using the Z-Score standardization.

- **Pipeline:** Combine all feature encoders into an unifed vector.

**Data Components**

- **Alpaca Client:** API client wrapper for ferching historical and real-time market data.

- **Persistence:** Lightweight JSON-inline lazy loading database for storing of data.

**Obervability and Monitoring**

- **Better Stack:** An observability and monitoring platform, that combine graphs, logs, incidents management, and machine learning.

## Cloud Service Managment

The cloud provider platform that uses this architecture is Amazon Web Services (AWS) with the Ubuntu Operating System. You can use Remote SSH to manage your server in Visual Studio Code (VSC). For preference use an AWS in the region of 'us-east-1', because the nearest and more important stock exchange is in New York.

```bash
# Update the package list to ensure the correct managment.
$ sudo apt update

# Install the last python3 aviable version and verify the installation.
$ sudo apt install python3
$ python3 --version

# Install the pip packages manager and the build essentials.
$ sudo apt install python3-pip build-essential

# Check if git us installed, and if not, install it.
$ sudo apt install git
```

## GitHub Private Repo Managment

```bash
# Clone the GitHub repository.
$ git clone https://github.com/dylan-sutton-chavez/aardvark-package.git
Username for 'https://github.com': your-github-user
Password for 'https://your-github-user@github.com': personal-acess-token

# Move to the cloned GitHub repository and create the file for the datasets.
$ cd aardvark-package
$ mkdir datasets
```

> How to Obtain the Personal Acess Token (PAC): GitHub Website → Settings → Developer Settings → Personal Acess Tokens → Tokens (Classic) → Generate New Token.

## Libraries Installation (Python: 3.13.9+)

- numpy: 2.3.4
- alpaca-py: 0.43.2
- psutil: 7.1.3

```bash
# Install the libraries of the system.
$ pip install -r $ requirements.txt
```

## Enviroment Initialization

```bash
# Initialize the Alpaca Client enviroment variables (Key and Secret).
$ export ALPACA_KEY="a1b2c3d4e5f6g7h8i9j0"
$ export ALPACA_SECRET="a1b2c3d4e5f6g7h8i9j0"

# Initialize the Better Stack enviroment variables (Source → Logs → Python).
$ export BETTER_STACK_HOST="a1b2c3d4e5f6g7h8i9j0"
$ export BETTER_STACK_TOKEN="a1b2c3d4e5f6g7h8i9j0"
```

## Persistence System Launching

```bash
# Allows the continious runing even after the terminal session is closed (run the program as a package '-m' and without 'byte-compile' -B).
$ nohup python3 -B -m main
```

## System Reboot Command

```bash
# If you whant to reboot the aardbark system (e.g., to update the software), you need to reboot the system and make again the repository, deleting the last launch.
$ sudo reboot
```

## Proprietary Agreement License

The software is licensed under a proprietary agreement, maintaining closed source distribution.



