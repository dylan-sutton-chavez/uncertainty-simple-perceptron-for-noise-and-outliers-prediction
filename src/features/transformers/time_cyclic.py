HOURS_IN_DAY: int = 24
MINUTES_IN_HOUR: int = 60
DAYS_IN_WEEK: int = 7
MONTHS_IN_YEAR: int = 12

from math import sin, cos, pi

class CyclicTime:
    def __init__(self, minute_in_hour: int, hour_in_day: int, day_in_week: int, month_in_year: int):
        """
        Initialize the object, where receive the minutes, hours, days, and months.
    
        Args:
            minute_in_hour: int → Minutes in hour to encode.
            hour_in_day: int → Hour in day to encode.
            day_in_week: int → Days in week to encode.
            month_in_year: int → Months in year to encode.

        Output:
            None

        Time complexity → O(1)
        """
        sin_hour, cos_hour = self._encode_time(minute_in_hour, MINUTES_IN_HOUR)
        sin_day, cos_day = self._encode_time(hour_in_day, HOURS_IN_DAY)
        sin_week, cos_week = self._encode_time(day_in_week, DAYS_IN_WEEK)

        adjusted_month: float = month_in_year - 1 # adjust the scale to avoid cyclic overlap
        sin_month, cos_month = self._encode_time(adjusted_month, MONTHS_IN_YEAR)

        self.encoded_features: dict[str, float] = {
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,

            'sin_day': sin_day,
            'cos_day': cos_day,

            'sin_week': sin_week,
            'cos_week': cos_week,

            'sin_month': sin_month,
            'cos_month': cos_month
        }

    def _encode_time(self, current_time: int, time_scale: int):
        """
        Encode the time computing the sin and cos.
    
        Args:
            current_time: int → Receive the current time.
            time_scale: int → The scale for the current time.

        Output:
            float → Compute sin of the radians. 
            float → Cos of the radians.

        Time complexity → O(1)
        """
        radians: float = self._calculate_radians(current_time, time_scale)
        return sin(radians), cos(radians)
    
    def _calculate_radians(self, value: int, period: int):
        """
        Calculate the radians whit a given value and period.
    
        Args:
            value: int → Value to calculate radians in the period window.
            period: int → Full period window.

        Output:
            None

        Time complexity → O(1)

        Maths:
            (2 * pi * (value)) / period)
        """
        return (2 * pi * (value)) / period
    
if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.features.transformers.time_cyclic
    """
    cyclic_time = CyclicTime(minute_in_hour=30, hour_in_day=3, day_in_week=1, month_in_year=1)
    print(cyclic_time.encoded_features)