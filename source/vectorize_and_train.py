class ModelHyperparameters:
    def __init__(self, epochs: int, patience: int, learning_rate: float):
        """
        Initialize the model hyperparametrs, for the model training.

        Args:
            epochs: int → Maximum number of full training cycles to run.
            patience: int → Epochs to wait before triggering early stopping.
            learning_rate: float → Controls how much the weights adjust during the training.

        Output:
            None

        Time complexity → O(1)
        """
        self.epochs: int = epochs
        self.patience: int = patience
        self.learning_rate: float = learning_rate

class MarketMetadata:
    def __init__(self, window_periods: int, symbol: str):
        """
        Start an object whit a given window preiods (in minutes), and the symbol of the market.

        Args:
            window_periods: int → Defines the window of time in weeks of the bars.
            symbol: str → Stock ticker symbol (e.g., 'TSLA') for which historical and latest market data.

        Output:
            None

        Time complexity → O(1)
        """
        self.window_periods: int = window_periods
        self.symbol: str = symbol

from apis.alpaca_markets import AlpacaMarkets

class AlpacaMarketsClient:
    def __init__(self, alpaca_key: str, alpaca_secret: str, symbol: str):
        """
        Initialize the alpaca merkets client with a given key, secret and symbol.

        Args:
            alpaca_key: str → API key needed for authenticating requests to the Alpaca Markets.
            alpaca_secret: str → Password that confirm and authorize access to the Alpaca API
            symbol: str → Stock ticker symbol (e.g., 'TSLA') for which historical and latest market data.

        Output:
            None

        Time complexity → O(1)
        """
        self.alpaca_markets_client = AlpacaMarkets(alpaca_key, alpaca_secret, symbol)

from core.uncertainty_simple_perceptron import UncertaintySimplePerceptron
from database.duck_db import DuckDB
from feature_encoders.z_score import ZScore
from features_vectorizer import TimeSeriesConfig, TimeConfig, NormalizationConfig, features_vectorizer

from datetime import datetime

class Train:
    def __init__(self, model_hyperparamets: ModelHyperparameters, market_metadata: MarketMetadata, alpaca_markets_client: AlpacaMarketsClient, core_model: UncertaintySimplePerceptron):
        """
        Initialize the training object, where can prepare a vectorized database of data, and train a given model object.

        Args:
            model_hyperparamets: ModelHyperparameters → Initialize the model hyperparametrs, for the model training.
            market_metadata: MarketMetadata → Start an object whit a given window preiods (in minutes), and the symbol of the market.
            alpaca_markets_client: AlpacaMarketsClient → Initialize the alpaca merkets client with a given key, secret and symbol.
            core_model: UncertaintySimplePerceptron → Object whit weights, bias, file paths, and metadata for the perceptron.

        Output:
            None

        Time complexity → O(1)
        """
        self.model_hyperparamets: ModelHyperparameters = model_hyperparamets
        self.market_metadata: MarketMetadata = market_metadata
        self.alpaca_markets_client: AlpacaMarkets = alpaca_markets_client.alpaca_markets_client

        self.core_model: UncertaintySimplePerceptron = core_model

        self.training_db = DuckDB(f'{market_metadata.symbol.lower()}_test.set')
        self.config_db = DuckDB(f'{market_metadata.symbol.lower()}_config.set')

        self.zscore_volume_obj: ZScore = None
        self.zscore_trade_count_obj: ZScore = None
        self.market_metadatazscore_vwap_obj: ZScore = None

    def prepare_database(self):
        """
        Prepare a database to train a model (receive, vectorize, and save in JSON format).

        Args:
            None

        Output:
            None
        """
        historical_market_bars: dict[str, any] = self.alpaca_markets_client.historical_market_bars(limit_bars=None, weeks_data_window=144)[self.market_metadata.symbol]

        self._compute_zscore_sets(historical_market_bars)

        training_set: list[dict] = []

        for index, bar in enumerate(historical_market_bars):
            window_periods: int = self.market_metadata.window_periods

            raw_window: list[dict] = historical_market_bars[index - window_periods: index]
            raw_prices_window: list[float] = [bar_dict.close for bar_dict in raw_window]

            if len(raw_prices_window) < window_periods:
                continue

            model_vectorization_dict: dict[str, any] = self._model_vectorization(bar, window_periods, raw_prices_window)

            entry_price = bar.close

            up_take_profit_price = entry_price + model_vectorization_dict['take_profit']
            up_stop_loss_price = entry_price - model_vectorization_dict['stop_loss']
            up_stop_loss_was_triggered = False

            down_take_profit_price = entry_price - model_vectorization_dict['take_profit']
            down_stop_loss_price = entry_price + model_vectorization_dict['stop_loss']
            down_stop_loss_was_triggered = False

            next_price_bars = historical_market_bars[index + 1: index + 1 + window_periods]
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
                'label': bar_label
            }

            training_set.append(vector)

        self.training_db.truncate_and_insert_list(training_set)

    def train_model(self):
        """
        Train a model whit the object database, using the given epochs, patience, and learning rate.

        Args:
            None

        Output:
            None
        """
        epochs: int = self.model_hyperparamets.epochs
        patience: int = self.model_hyperparamets.patience
        learning_rate: float = self.model_hyperparamets.learning_rate

        dataset_name: str = self.training_db.database_file_name

        self.core_model.train(dataset_name, epochs, patience, learning_rate, save_model=False)

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

        zscore_config = {
            'volume_means': self.zscore_volume_obj.means,
            'volume_std': self.zscore_volume_obj.std,
            
            'trade_count_means': self.zscore_trade_count_obj.means,
            'trade_count_std': self.zscore_trade_count_obj.std,

            'vwap_means': self.zscore_vwap_obj.means,
            'vwap_std': self.zscore_vwap_obj.std
        }

        self.config_db.truncate()
        self.config_db.insert(zscore_config)

    def _model_vectorization(self, bar: dict[str, any], window_periods: int, raw_prices_window: list[float]):
        """
        Receive the current market bar, and compute the whole vector, stop loss, time stamp, and take profit.

        Args:
            bar: dict[str, any] → Current market bar to compute.
            window_periods: int → A integer defining the length of the market window.
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
        
        time_series_config = TimeSeriesConfig(raw_prices_window, window_periods)
        time_config = TimeConfig(bar_timestamp.minute + 1, bar_timestamp.hour + 1, bar_timestamp.weekday() + 1, bar_timestamp.month)
        normalization_config = NormalizationConfig(self.zscore_volume_obj, bar_volume, self.zscore_trade_count_obj, bar_trade_count, self.zscore_vwap_obj, bar_vwap)

        features_vector = features_vectorizer(time_series_config, time_config, normalization_config)

        return {'take_profit': take_profit, 'stop_loss': stop_loss, 'bar_formated_timestamp': bar_formated_timestamp, 'features_vector': features_vector}

if __name__ == '__main__':
    """
    Run a training database preparation, and train a model with the database.

    Root (cd ../aardvark-uncertainty_simple_perceptron):
        python -B -m source.vectorize_and_train
    """
    EPOCHS: int = 170
    PATIENCE: int = 20
    LEARNING_RATE: float = 0.0001

    model_hyperparameters = ModelHyperparameters(EPOCHS, PATIENCE, LEARNING_RATE)

    WINDOW_PERIODS: int = 14
    SYMBOL: str = 'TSLA'

    market_metadata = MarketMetadata(WINDOW_PERIODS, SYMBOL)

    ALPACA_KEY = '1234567890'
    ALPACA_SECRET = '1234567890'

    alpaca_markets_client = AlpacaMarketsClient(ALPACA_KEY, ALPACA_SECRET, SYMBOL)

    NAME: str = 'Tesla Stocks Uncertainty Simple Perceptron HFT'
    DESCRIPTION: str = 'Uncertainty Simple Perceptron Model, for High-Frequency Trading for Tesla (risk 3:1 and 14 periods of 15 minutes).'
    AUTHOR: str = 'Dylan Sutton Chavez'

    core_model = UncertaintySimplePerceptron(NAME, DESCRIPTION, AUTHOR)

    train = Train(model_hyperparameters, market_metadata, alpaca_markets_client, core_model)

    train.prepare_database()

    train.train_model()
