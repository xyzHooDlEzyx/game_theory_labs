from fractions import Fraction
from pathlib import Path
import matplotlib.pyplot as plt


c = [
    [-4, 5],
    [4, -3],
    [-1, -2],
    [0, -1],
    [2, 1],
    [3, -4],
]


def brown_robinson(matrix, iterations=10, start_a=1, start_b=1):
    """Computes the approximate solution of a two-player zero-sum game using the Brown-Robinson method."""
    row_count = len(matrix)
    column_count = len(matrix[0])

    counts_a = [0] * row_count
    counts_b = [0] * column_count
    strategy_a = start_a - 1
    strategy_b = start_b - 1
    results = []

    for number in range(1, iterations + 1):
        counts_a[strategy_a] += 1
        counts_b[strategy_b] += 1

        probabilities_a = [Fraction(count, number) for count in counts_a]
        probabilities_b = [Fraction(count, number) for count in counts_b]

        gains_a = []
        for row in range(row_count):
            gain = sum(
                Fraction(matrix[row][column]) * probabilities_b[column]
                for column in range(column_count)
            )
            gains_a.append(gain)

        losses_b = []
        for column in range(column_count):
            loss = sum(
                Fraction(matrix[row][column]) * probabilities_a[row]
                for row in range(row_count)
            )
            losses_b.append(loss)

        alpha = max(gains_a)
        beta = min(losses_b)
        value = (alpha + beta) / 2

        results.append(
            {
                "number": number,
                "strategy_a": strategy_a + 1,
                "strategy_b": strategy_b + 1,
                "probabilities_a": probabilities_a,
                "probabilities_b": probabilities_b,
                "alpha": alpha,
                "beta": beta,
                "value": value,
            }
        )

        strategy_a = gains_a.index(alpha)
        strategy_b = losses_b.index(beta)

    return results


def format_number(value):
    """Returns a string representation of a Fraction, either as an integer or as a fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_vector(values):
    """Formats a vector of probabilities for display."""
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def print_results(results):
    print(" N  iN  jN    alphaN     betaN        vN")
    print("-" * 49)

    for item in results:
        print(
            f"{item['number']:2d}  "
            f"{item['strategy_a']:2d}  "
            f"{item['strategy_b']:2d}  "
            f"{format_number(item['alpha']):>8}  "
            f"{format_number(item['beta']):>8}  "
            f"{format_number(item['value']):>8}"
        )


def plot_results(results):
    """Builds a plot of the approximate game value as a function of the iteration number."""

    iterations = [item["number"] for item in results]
    values = [float(item["value"]) for item in results]

    plt.figure(figsize=(9, 5))
    plt.plot(iterations, values, marker="o", label="approximate value vN")
    plt.axhline(1.4, color="red", linestyle="--", label="exact v = 1,4")
    plt.xticks(iterations)
    plt.xlabel("Iteration N")
    plt.ylabel("Approximate game value vN")
    plt.title("Dependency of the approximate game value on the iteration number")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    results = brown_robinson(c, iterations=10)
    print_results(results)

    last = results[-1]
    print("\nEmpirical strategy of player A:", format_vector(last["probabilities_a"]))
    print("Empirical strategy of player B:", format_vector(last["probabilities_b"]))
    print("Approximate game value:", format_number(last["value"]))
    plot_results(results)


if __name__ == "__main__":
    main()
