from config.settings import NAME, DESCRIPTION, AUTHOR, WINDOW_PERIODS, MARKET_SYMBOL

from src.data.alpaca_client import AlpacaMarkets
from src.model.perceptron import UncertaintySimplePerceptron
from src.data.persistence.json_store import JsonFileStore
from src.features.transformers.zscore_scaler import ZScore
from src.features.pipeline import TimeSeriesConfig, TimeConfig, NormalizationConfig, features_vectorizer

from datetime import datetime

class VectorizeAndTrain:
    def __init__(self, alpaca_markets_client: AlpacaMarkets):
        """
        Initialize the training object, where can prepare a vectorized database of data, and train a given model object.

        Args:
            alpaca_markets_client: AlpacaMarkets → Initialize the alpaca merkets client with a given key, secret and symbol.

        Output:
            None

        Time complexity → O(1)
        """
        self.alpaca_markets_client: AlpacaMarkets = alpaca_markets_client
        self.core_model: UncertaintySimplePerceptron = UncertaintySimplePerceptron(NAME, DESCRIPTION, AUTHOR)

        self.training = JsonFileStore(f"{MARKET_SYMBOL.lower()}_training")
        self.training.truncate()

        self.zscore_volume_obj: ZScore = None
        self.zscore_trade_count_obj: ZScore = None
        self.zscore_vwap_obj: ZScore = None

    def prepare_database(self):
        """
        Prepare a database to train a model (receive, vectorize, and save in JSON format).

        Args:
            None

        Output:
            None
        """
        historical_market_bars: dict[str, any] = self.alpaca_markets_client.historical_market_bars(limit_bars=None, weeks_data_window=144)[MARKET_SYMBOL]

        self._compute_zscore_sets(historical_market_bars)

        for index, bar in enumerate(historical_market_bars):

            raw_window: list[dict] = historical_market_bars[index - WINDOW_PERIODS: index]
            raw_prices_window: list[float] = [bar_dict.close for bar_dict in raw_window]

            if len(raw_prices_window) < WINDOW_PERIODS:
                continue

            model_vectorization_dict: dict[str, any] = self._model_vectorization(bar, raw_prices_window)

            entry_price = bar.close

            up_take_profit_price = entry_price + model_vectorization_dict['take_profit']
            up_stop_loss_price = entry_price - model_vectorization_dict['stop_loss']
            up_stop_loss_was_triggered = False

            down_take_profit_price = entry_price - model_vectorization_dict['take_profit']
            down_stop_loss_price = entry_price + model_vectorization_dict['stop_loss']
            down_stop_loss_was_triggered = False

            next_price_bars = historical_market_bars[index + 1: index + 1 + WINDOW_PERIODS]
            bar_label = 0.5

            for next_bar in next_price_bars:

                if next_bar.high <= up_stop_loss_price:
                    up_stop_loss_was_triggered = True

                if next_bar.high >= up_take_profit_price and up_stop_loss_was_triggered == False:
                    bar_label = 1
                    break

                if next_bar.low >= down_stop_loss_price:
                    down_stop_loss_was_triggered = True

                if next_bar.low <= down_take_profit_price and down_stop_loss_was_triggered == False:
                    bar_label = 0
                    break

            if bar_label == 0.5:
                continue

            vector = {
                'timestamp': model_vectorization_dict['bar_formated_timestamp'],
                'vector': model_vectorization_dict['features_vector'],
                'target': bar_label
            }

            self.training.insert(vector)

    def train_model(self):
        """
        Train a model whit the object database, using the given epochs, patience, and learning rate.

        Args:
            None

        Output:
            None
        """
        self.core_model.train(self.training, save_model=False)

    def vectorized_last_window_bars(self):
        """
        Access to the last window bars, and compute a vector using the model vectorization.
        
        Args:
            None
            
        Output:
            dict[str, any] → A dictionary returning the take profit, stop loss, time stamp, and the features vector.
            float → The close price of the last bar.
        """
        raw_window = self.alpaca_markets_client.last_window_bars()[MARKET_SYMBOL]
        raw_prices_window: list[float] = [bar_dict.close for bar_dict in raw_window]
        bar = raw_window[-1]

        return self._model_vectorization(bar, raw_prices_window), bar.close

    def _compute_zscore_sets(self, historical_market_bars: list[str, any]):
        """
        Receive a given set for the historical maket price and compute the Z-score for volume, trade count, and vwap.
        
        Args:
            historical_market_bars: list[str, any] → List of market bars with dictionary information.
            
        Output:
            None
        """
        self.zscore_volume_obj = ZScore([bar.volume for bar in historical_market_bars])
        self.zscore_trade_count_obj = ZScore([bar.trade_count for bar in historical_market_bars])
        self.zscore_vwap_obj = ZScore([bar.vwap for bar in historical_market_bars])

    def _model_vectorization(self, bar: dict[str, any], raw_prices_window: list[float]):
        """
        Receive the current market bar, and compute the whole vector, stop loss, time stamp, and take profit.

        Args:
            bar: dict[str, any] → Current market bar to compute.
            raw_prices_window: list[float] → One vector with the window prices.

        Output:
            dict[str, any] → A dictionary returning the take profit, stop loss, time stamp, and the features vector
        """
        price_window_range: float = bar.high - bar.low

        stop_loss: float = 0.5 * price_window_range
        take_profit: float = 1.5 * price_window_range

        bar_timestamp: datetime = bar.timestamp
        bar_formated_timestamp: str = bar_timestamp.strftime('%Y-%m-%d %H:%M %Z')

        bar_volume: float = bar.volume
        bar_trade_count: int = bar.trade_count
        bar_vwap: float = bar.vwap
        
        time_series_config = TimeSeriesConfig(raw_prices_window, WINDOW_PERIODS)
        time_config = TimeConfig(bar_timestamp.minute + 1, bar_timestamp.hour + 1, bar_timestamp.weekday() + 1, bar_timestamp.month)
        normalization_config = NormalizationConfig(self.zscore_volume_obj, bar_volume, self.zscore_trade_count_obj, bar_trade_count, self.zscore_vwap_obj, bar_vwap)

        features_vector = features_vectorizer(time_series_config, time_config, normalization_config)

        return {'take_profit': take_profit, 'stop_loss': stop_loss, 'bar_formated_timestamp': bar_formated_timestamp, 'features_vector': features_vector}

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m scripts.vectorize_and_train
    """
    from config.settings import MARKET_SYMBOL
    
    from os import getenv

    alpaca_markets_client = AlpacaMarkets(getenv('ALPACA_KEY'), getenv('ALPACA_SECRET'), MARKET_SYMBOL)

    vectorize_and_train = VectorizeAndTrain(alpaca_markets_client)

    vectorize_and_train.prepare_database()
    vectorize_and_train.train_model()

    model = vectorize_and_train.core_model

    vectorized_last_window_bars, _ = vectorize_and_train.vectorized_last_window_bars()
    print(vectorized_last_window_bars)