from numpy import array, diff, mean, maximum, minimum

class IndicatorCalculator:
    def __init__(self, raw_vector: list[float], window_periods: int):
        """
        Initialize the 'IndicatorCalculator', and assemble a vector with a given raw vector of prices and the window periods window length.

        Args:
            raw_vector: list[float] → List of closing prices.
            window_periods: int → Integer representing the length of the time window to use.

        Output:
            None

        Time complexity → O(n)
        """
        self.raw_vector = raw_vector
        self.raw_vector_window = raw_vector[-window_periods:]

        self.periods = window_periods

        self.processed: dict[str, float] = {
                "stochastic_oscillator": self._stochastic_oscillator(),
                "relative_strength_index": self._relative_strength_index(),
                "exponential_moving_average": self._exponential_moving_average()
            }

    def _stochastic_oscillator(self):
        """
        Momentum indicator within technical analysis that uses support and resistance levels as an oscillator.
        
        Args:
            None

        Output:
            float → Relative position of the last price inside the range of values in the period. 

        Time complexity → O(n)

        Maths:
            (Price - Low) / (High - Low)
        """
        if not self.raw_vector_window:
            return 0.0
        
        current: float = self.raw_vector_window[-1]
        
        higher: float = max(self.raw_vector_window)
        lower: float = min(self.raw_vector_window)

        difference: float = higher - lower

        if difference == 0.0:
            return 0.0
        
        return (current - lower) / difference
    
    def _relative_strength_index(self):
        """
        Chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period.

        Args:
            None

        Output:
            float → Momentum indicator for the strength of the movements scaled from 0 to 1.

        Time complexity → O(n)

        Maths:
            1 - (1 / (1 + (total_gain / total_loss)))
        """
        if len(self.raw_vector_window) < 2:
            return 0.5

        vector = array(self.raw_vector_window)
        differences = diff(vector)

        total_gain = sum(maximum(differences, 0))
        total_loss = sum(minimum(differences, 0)) * -1

        if total_loss == 0:
            return 1.0 if total_gain > 0 else 0.5
        
        relative_strength = total_gain / total_loss

        return float(1 - (1 / (1 + relative_strength)))
    
    def _exponential_moving_average(self):
        """
        Smooth time series data using an exponential window, assigning exponentially decreasing weights to past observations over time.

        Args:
            None

        Output:
            float → Smoother time series indicator in the range from 0 to 1

        Time complexity → O(n)

        Maths:
            multiplier = 2 / (periods + 1)
            (feature * multiplier) + (moving_average * (1 - multiplier))
        """
        if len(self.raw_vector_window) < self.periods:
            return 0.0
        
        multiplier: float = 2.0 / (self.periods + 1.0)

        # Initialize 'moving_average' calculating the simple moving average (mean)
        moving_average: float = mean(self.raw_vector_window)
        
        values: list[float] = [moving_average]

        for feature in self.raw_vector_window[1:]:
            moving_average += multiplier * (feature - moving_average)
            values.append(moving_average)

        maximum: float = max(values)
        minimum: float = min(values)

        range = maximum - minimum

        if range == 0:
            return 1.0

        return float((moving_average - minimum) / range)

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.features.transformers.indicators
    """
    from config.settings import WINDOW_PERIODS

    raw_vector: list[float] = [107314.99, 107046.21, 106881.76, 106847.01, 106401.76, 106368.61, 106338.20, 106296.88, 106191.95, 106180.25, 106096.38, 105990.87, 105822.84, 105786.13, 105655.45, 105634.34, 105574.92, 105494.67, 105342.13, 105307.74, 105260.60, 105228.61, 105221.78, 105206.15, 104981.18, 104941.50, 104919.86, 104881.09, 104764.00, 104753.86, 104711.23, 104618.30, 104599.55, 104576.63, 104449.65, 104280.00, 104000.00, 103900.00, 104100.00, 104200.00, 104400.00, 104150.00, 104050.00, 103800.00, 103600.00, 103750.00, 104125.00, 104250.00, 104180.00]

    indicator_calculator = IndicatorCalculator(raw_vector, WINDOW_PERIODS)
    print(indicator_calculator.processed)