from src.features.transformers.indicators import IndicatorCalculator
from src.features.transformers.time_cyclic import CyclicTime
from src.features.transformers.zscore_scaler import ZScore

class TimeSeriesConfig:
    def __init__(self, raw_vector: list[float], periods: int):
        """
        Initializes time series config with raw data vector and periods.

        Args:
            raw_vector: list[float] → List of floats representing the raw time series data.
            periods: int → Number of lookback periods for technical indicator calculation.

        Output:
            None

        Time complexity → O(1)
        """
        self.raw_vector: list[float] = raw_vector
        self.periods: int = periods

class TimeConfig:
    def __init__(self, current_minute: int, current_hour: int, current_day_of_week: int, current_month: int):
        """
        Initializes time config with minute, hour, day, and month.

        Args:
            current_minute: int → The current minute of the hour for cyclic encoding.
            current_hour: int → The current hour of the day for cyclic encoding.
            current_day_of_week: int → The current day of the week for cyclic encoding.
            current_month: int → The current month of the year for cyclic encoding.

        Output:
            None

        Time complexity → O(1)
        """
        self.current_minute: int = current_minute
        self.current_hour: int = current_hour
        self.current_day_of_week: int = current_day_of_week
        self.current_month: int = current_month

class NormalizationConfig:
    def __init__(self, volume_zscore_obj: ZScore, volume: float, trade_count_zscore_obj: ZScore, trade_count: float, vwap_zscore_obj: ZScore, vwap: float):
        """
        Initializes ZScore objects and raw values for normalization config.

        Args:
            volume_zscore_obj: ZScore → An object to calculate the Z-score for trading volume.
            volume: float → The current raw trading volume value to be normalized.
            trade_count_zscore_obj: ZScore → An object to calculate the Z-score for trade count.
            trade_count: float → The current raw trade count value to be normalized.
            vwap_zscore_obj: ZScore → An object to calculate the Z-score for the VWAP.
            vwap: float → The current raw Volume-Weighted Average Price value to normalize.

        Output:
            None

        Time complexity → O(1)
        """
        self.volume_zscore_obj: ZScore = volume_zscore_obj
        self.volume: float = volume
        self.trade_count_zscore_obj: ZScore = trade_count_zscore_obj
        self.trade_count: float = trade_count
        self.vwap_zscore_obj: ZScore = vwap_zscore_obj
        self.vwap: float = vwap

def features_vectorizer(time_series_config: TimeSeriesConfig, time_config: TimeConfig, normalization_config: NormalizationConfig):
    """
    This function vectorizes time series features, cyclic time encodings...
    
    Args:
        time_series_config: TimeSeriesConfig
        time_config: TimeConfig
        normalization_config: NormalizationConfig

    Output:
        list[float] → Function returns a list of floating point numbers, representaiting each feature.

    Time complexity → O(n1 * m * c) * (n2)
    """
    time_series_features = IndicatorCalculator(time_series_config.raw_vector, time_series_config.periods)
    cyclic_time_encoder = CyclicTime(time_config.current_minute, time_config.current_hour, time_config.current_day_of_week, time_config.current_month)

    technical_indicators = time_series_features.processed
    cyclic_time = cyclic_time_encoder.encoded_features

    vector: list[float] = [
            technical_indicators['stochastic_oscillator'], 
            technical_indicators['relative_strength_index'], 
            technical_indicators['exponential_moving_average'],

            cyclic_time['sin_hour'],
            cyclic_time['cos_hour'],
            cyclic_time['sin_day'],
            cyclic_time['cos_day'],
            cyclic_time['sin_week'],
            cyclic_time['cos_week'],
            cyclic_time['sin_month'],
            cyclic_time['cos_month'],

            normalization_config.volume_zscore_obj.compute_z_score(normalization_config.volume),
            normalization_config.trade_count_zscore_obj.compute_z_score(normalization_config.trade_count),
            normalization_config.vwap_zscore_obj.compute_z_score(normalization_config.vwap)
        ]

    return vector

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.features.pipeline
    """
    raw_vector: list[float] = [107314.99, 107046.21, 106881.76, 106847.01, 106401.76, 106368.61, 106338.20, 106296.88, 106191.95, 106180.25, 106096.38, 105990.87, 105822.84, 105786.13, 105655.45, 105634.34, 105574.92, 105494.67, 105342.13, 105307.74, 105260.60, 105228.61, 105221.78, 105206.15, 104981.18, 104941.50, 104919.86, 104881.09, 104764.00, 104753.86, 104711.23, 104618.30, 104599.55, 104576.63, 104449.65, 104280.00, 104000.00, 103900.00, 104100.00, 104200.00, 104400.00, 104150.00, 104050.00, 103800.00, 103600.00, 103750.00, 104125.00, 104250.00, 104180.00]
    periods: int = 14

    current_minute: int = 23
    current_hour: int = 23
    current_day_of_week: int = 1
    current_month: int = 1 

    volume_vector: list[float] = [6332194.0, 9863011.0, 7096410.0, 7582205.0, 8897391.0, 7218738.0, 5385225.0, 5196582.0, 4792414.0, 7698978.0, 5864020.0, 5876154.0, 7384279.0, 4486704.0, 224497.0, 72071.0, 55582.0, 85630.0, 64354.0, 49626.0, 333968.0, 307390.0, 115719.0, 84793.0, 133077.0, 220933.0, 127448.0, 46489.0, 81775.0, 52735.0]
    volume_zscore_obj: ZScore = ZScore(volume_vector)
    volume: float = 7096410.0

    trade_count_vector: list[float] = [50238.0, 77138.0, 45948.0, 55542.0, 66207.0, 46806.0, 36941.0, 37613.0, 35417.0, 57036.0, 41743.0, 42421.0, 64524.0, 3533.0, 2189.0, 1465.0, 1170.0, 1038.0, 1059.0, 1070.0, 3244.0, 6827.0, 3187.0, 2522.0, 3840.0, 4954.0, 3133.0, 1574.0, 2186.0, 1389.0]
    trade_count_zscore_obj: ZScore = ZScore(trade_count_vector)
    trade_count: float = 42421.0

    vwap_vector: list[float] = [195.859472, 192.699425, 192.26061, 190.886518, 190.833788, 192.0536, 191.109232, 191.042444, 190.543425, 189.478978, 190.516543, 190.172993, 190.096023, 189.987335, 190.089917, 190.130903, 190.090525, 190.051154, 190.02602, 189.955971, 191.125625, 193.192167, 193.452718, 193.929741, 192.841773, 191.289834, 191.170514, 191.43263, 191.068376, 191.10855]
    vwap_zscore_obj: ZScore = ZScore(vwap_vector)
    vwap: float = 190.130903

    time_series_config: TimeSeriesConfig = TimeSeriesConfig(raw_vector=raw_vector, periods=periods)
    time_config: TimeConfig = TimeConfig(current_minute=current_minute, current_hour=current_hour, current_day_of_week=current_day_of_week, current_month=current_month)
    normalization_config: NormalizationConfig = NormalizationConfig(volume_zscore_obj=volume_zscore_obj, volume=volume, trade_count_zscore_obj=trade_count_zscore_obj, trade_count=trade_count, vwap_zscore_obj=vwap_zscore_obj, vwap=vwap)

    vector = features_vectorizer(time_series_config, time_config, normalization_config)
    print(vector)