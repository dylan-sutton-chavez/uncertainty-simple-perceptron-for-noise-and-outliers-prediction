from config.settings import MARKET_SYMBOL, EPSILON

from src.data.alpaca_client import AlpacaMarkets
from src.model.perceptron import UncertaintySimplePerceptron

from scripts.vectorize_and_train import VectorizeAndTrain

from alpaca.trading.client import TradingClient

from time import perf_counter
from os import getenv
from datetime import datetime, UTC
from psutil import cpu_percent, virtual_memory
from requests import post
from json import dumps

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

class AardvarkObject:
    def __init__(self):
        """
        Initialize the 'AardvarkObject' object in memory, and start all the enviroment keys, clients and the model.

        Args:
            None

        Output:
            None
        """
        alpaca_key: str = getenv('ALPACA_KEY')
        alpaca_secret: str = getenv('ALPACA_SECRET')

        self.better_stack_host: str = getenv('BETTER_STACK_HOST')
        self.better_stack_token: str = getenv('BETTER_STACK_TOKEN')

        self.alpaca_trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True) # You can replaca the paper trading from True or False.

        alpaca_markets_client = AlpacaMarkets(alpaca_key, alpaca_secret, MARKET_SYMBOL)
        self.vectorize_and_train = VectorizeAndTrain(alpaca_markets_client)

        self.train_model()

    def run(self):
        """
        This function initialize and run entire inference and loging system.

        Args:
            None

        Output:
            None
        """
        start: float = perf_counter()

        vectorized_last_window_bars_return = self.vectorize_and_train.vectorized_last_window_bars()

        if vectorized_last_window_bars_return is None:
            print("The data is missing.")
            return

        vectorized_last_window_bars: dict[str, any] = vectorized_last_window_bars_return['vector']
        last_bar_close_price: float = vectorized_last_window_bars_return['last_close_price']

        pred, net_pred = self.model.inference(vectorized_last_window_bars['features_vector'], EPSILON)

        if pred == 1:
            order = MarketOrderRequest(
                symbol=MARKET_SYMBOL, 
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(last_bar_close_price + vectorized_last_window_bars['take_profit'], 2)),
                stop_loss=StopLossRequest(stop_price=round(last_bar_close_price - vectorized_last_window_bars['stop_loss'], 2))
            )
            self.alpaca_trading_client.submit_order(order)

        if pred == 0:
            order = MarketOrderRequest(
                symbol=MARKET_SYMBOL, 
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(last_bar_close_price - vectorized_last_window_bars['take_profit'], 2)),
                stop_loss=StopLossRequest(stop_price=round(last_bar_close_price + vectorized_last_window_bars['stop_loss'], 2))
            )
            self.alpaca_trading_client.submit_order(order)

        latency: float = perf_counter() - start

        model_log = {
            "timestamp": str(datetime.now(UTC)),
            "latency_seconds": round(latency, 7),
            "symbol": MARKET_SYMBOL,
            "model_prediction": pred,
            "net_prediction": round(net_pred, 7),
            "os_health": {
                "cpu_load": cpu_percent(0.5),
                "cache_memmory": virtual_memory().percent
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.better_stack_token}"
        }

        post(self.better_stack_host, headers=headers, data=dumps(model_log), timeout=4)

    def train_model(self):
        """
        Prepare the database, train the model and load in the cache.

        Args:
            None

        Output:
            None
        """
        self.vectorize_and_train.prepare_database()
        self.vectorize_and_train.train_model()

        self.model: UncertaintySimplePerceptron = self.vectorize_and_train.core_model

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m main
    """
    from time import sleep
    
    execution_minutes: list[int] = [0, 15, 30, 45]
    aardvark_object = AardvarkObject()

    while True:

        current_time = datetime.now(UTC)
        minute = current_time.minute

        if minute in execution_minutes:

            aardvark_object.run()
            sleep(90) # Whait nineteen seconds to prevent double inference.

            aardvark_object.train_model()

        sleep(0.001)
