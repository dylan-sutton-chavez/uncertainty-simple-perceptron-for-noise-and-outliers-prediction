from config.settings import MINUTES_WINDOW, WINDOW_PERIODS

from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit

from datetime import datetime, timedelta, UTC

class AlpacaMarkets:
    def __init__(self, alpaca_key: str, alpaca_secret: str, symbol: str):
        """
        Initializes the data client using credentials and sets the specific stock symbol for future use.
        
        Args:
            alpaca_key: str → API key needed for authenticating requests to the Alpaca Markets.
            alpaca_secret: str → Password that confirm and authorize access to the Alpaca API
            symbol: str → Stock ticker symbol (e.g., 'AAPL') for which historical and latest market data.

        Output: 
            None

        Time complexity → o(1)
        """
        self.client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.symbol = symbol

    def historical_market_bars(self, limit_bars: int = None, weeks_data_window: int = 4):
        """
        Retrieves 15-minute interval historical price data for the specified symbol over a defined time period.
        
        Args:
            limit_bars: int → Maximum number of n-minutes price bars to retrieve
            weeks_data_window: int = 4 → Defines the window of time in weeks of the bars.

        Output: 
            None

        Time complexity → o(n)
        """
        now_time_utc = datetime.now(UTC)

        from_date = now_time_utc - timedelta(weeks=weeks_data_window)

        timeframe_of_n_minutes = TimeFrame(amount=MINUTES_WINDOW, unit=TimeFrameUnit.Minute)

        request_params = StockBarsRequest(symbol_or_symbols=self.symbol, timeframe=timeframe_of_n_minutes, start=from_date, end=now_time_utc, limit=limit_bars)

        return self.client.get_stock_bars(request_params)

    def last_window_bars(self):
        """
        Fetches the most recent, currently available market price window bars data for the initialized stock symbol.
        
        Args:
            None

        Output: 
            None

        Time complexity → o(1)
        """
        timeframe_of_n_minutes = TimeFrame(amount=MINUTES_WINDOW, unit=TimeFrameUnit.Minute)

        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol, 
            timeframe=timeframe_of_n_minutes, 
            limit=WINDOW_PERIODS
        )

        return self.client.get_stock_bars(request_params)
    
if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.data.alpaca_client
    """
    from os import getenv

    from config.settings import MARKET_SYMBOL
    
    alpaca_markets = AlpacaMarkets(getenv('ALPACA_KEY'), getenv('ALPACA_SECRET'), MARKET_SYMBOL)

    historical_market_bars = alpaca_markets.historical_market_bars()
    print(historical_market_bars)

    last_window_bars = alpaca_markets.last_window_bars()
    print(last_window_bars)