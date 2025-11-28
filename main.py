"""
This package is the main orchestrator of the 'aardvark' system. 
It handles two operating cycles: real-time inference and periodic retraining using multithreading and async processes.

In its inference cycle, the monitor fetches market data (the 'WINDOW_PERIODS' window) every 'MINUTES_WINDOW'. 
It loads the cached model, vectorizes the window data, and executes a prediction.

Every 'n' days, the monitor runs the retraining package (vectorize_and_train). 
This process ingests new historical data and adjusts the model to current market conditions.

The system use an external monitoring system (for: service health, statiscal data,...) its sendit to an external
observavility plataform such as: dynatrace, New Relic, DataDog,...
"""
from src.data.alpaca_client import AlpacaMarkets
from src.model.perceptron import UncertaintySimplePerceptron

from scripts.vectorize_and_train import VectorizeAndTrain

from config.settings import MARKET_SYMBOL, EPSILON

from time import perf_counter
from os import getenv
from datetime import datetime, UTC
from psutil import cpu_percent, virtual_memory
from requests import post
from json import dumps

alpaca_markets_client = AlpacaMarkets(getenv('ALPACA_KEY'), getenv('ALPACA_SECRET'), MARKET_SYMBOL)

better_stack_host: str = getenv('BETTER_STACK_HOST')
better_stack_token: str = getenv('BETTER_STACK_TOKEN')

def run_system():
    start_ms: float = perf_counter()

    vectorize_and_train = VectorizeAndTrain(alpaca_markets_client)

    vectorize_and_train.prepare_database()
    vectorize_and_train.train_model()

    model: UncertaintySimplePerceptron = vectorize_and_train.core_model
    vectorized_last_window_bars = vectorize_and_train.vectorized_last_window_bars()

    pred, net_pred = model.inference(vectorized_last_window_bars['features_vector'], EPSILON)

    latency_ms: float = perf_counter() - start_ms

    model_log = {
        "timestamp": str(datetime.now(UTC)),
        "latency_s": latency_ms / 1000,
        "symbol": MARKET_SYMBOL,
        "model_prediction": pred,
        "net_prediction": net_pred,
        "os_health": {
            "cpu_load": cpu_percent(0.5),
            "cache_memmory": virtual_memory().percent
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {better_stack_token}"
    }

    post(better_stack_host, headers=headers, data=dumps(model_log), timeout=4)

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m main
    """
    from config.settings import MINUTES_WINDOW

    from time import sleep
    
    execution_minutes: list[int] = [0, 15, 30, 45]

    while True:

        current_time = datetime.now(UTC)
        minute = current_time

        if minute in execution_minutes:

            run_system()
            sleep(60 * (MINUTES_WINDOW - 1)) # Wait a few minutes to avoid overloading the CPU.

        sleep(0.001)