class ZScore:
    def __init__(self, vector: list[float] = None, means: float = None, std: float = None):
        """
        Initialize the object, with a vector and compute the mean and standar desviation.

        Args:
            vector: list[float] = None → Vector of values to calculate the mean and standar desviation.
            means: float = None → Intermediate to the extreme values of the set of numbers.
            std: float = None → Measure of the amount of variation of the values of a variable around its mean.

        Output: 
            None

        Time complexity → O(n)
        """
        if not vector is None:
            self.means = self._mean(vector)
            self.std = self._standard_deviation(vector, self.means)
            
        if means is not None and std is not None:
            self.menas = means
            self.std = std

    def compute_z_score(self, value: float):
        """
        Standardizes variables to the same scale, producing new variables with a mean of 0 and a standard deviation of 1.

        Args:
            value: float → Value to compute using the Z-Score.

        Output:
            float → Standar puntuation of the value.

        Time complexity → O(1)

        Maths:
            (x - mean) / standar_desviation
        """
        return (value - self.means) / self.std

    def _mean(self, vector: list[float]):
        """
        Quantity representing the "center" of a collection of numbers and is intermediate to the extreme values of the set of numbers.

        Args:
            vector: list[float] → Vector of values to calculate the mean.

        Output:
            float → Intermediate to the extreme values of the set of numbers.

        Time complexity → O(n)

        Maths:
            sum(vector) / n
        """
        return sum(vector) / len(vector)

    def _standard_deviation(self, vector: list[float], mean: float):
        """
        The standard deviation is a measure of the amount of variation of the values of a variable about its mean.

        Args:
            vector: list[float] → Vector of values to calculate the variation of the data.
            mean: float → Mean of the given vector.

        Output:
            float → Measure of the amount of variation of the values of a variable around its mean.

        Time complexity → O(n)

        Maths:
            sqrt((sum(x - mean) ^ 2) / n)
        """
        square_sum = sum((floatant - mean) ** 2 for floatant in vector)
        return (square_sum / (len(vector) - 1)) ** 0.5

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.features.transformers.zscore_scaler
    """
    vector: list[float] = [1, 2, 3, 4]

    z_score = ZScore(vector)
    computed_z_scores: list[float] = [z_score.compute_z_score(value) for value in vector]

    print(computed_z_scores)