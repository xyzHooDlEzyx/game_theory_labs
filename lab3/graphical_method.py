from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt


MATRICES = {
    "A (2x4, saddle point)": [
        [10, 12, 14, 15],
        [8, 9, 11, 13],
    ],
    "B (2x4, no saddle point)": [
        [10, 15, 14, 13],
        [15, 10, 13, 14],
    ],
    "C (5x2, saddle point)": [
        [10, 15],
        [9, 14],
        [8, 13],
        [7, 12],
        [6, 11],
    ],
    "D (5x2, no saddle point)": [
        [10, 15],
        [15, 10],
        [11, 12],
        [12, 11],
        [9, 9],
    ],
}


def line_value(start, end, probability):
    """Returns the value on a line between its endpoint payoffs."""
    return Fraction(start) * probability + Fraction(end) * (1 - probability)


def intersection(first_start, first_end, second_start, second_end):
    """Returns the probability at which two payoff lines intersect."""
    first_slope = Fraction(first_start) - Fraction(first_end)
    second_slope = Fraction(second_start) - Fraction(second_end)
    denominator = first_slope - second_slope
    if denominator == 0:
        return None
    return (Fraction(second_end) - Fraction(first_end)) / denominator


def candidate_probabilities(lines):
    """Builds all endpoint and pairwise-intersection candidates on [0, 1]."""
    candidates = {Fraction(0), Fraction(1)}
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            point = intersection(*lines[first], *lines[second])
            if point is not None and 0 <= point <= 1:
                candidates.add(point)
    return sorted(candidates)


def pure_prices(matrix):
    """Returns row minima, column maxima, maximin, minimax, and saddle points."""
    row_minimums = [min(row) for row in matrix]
    column_maximums = [max(column) for column in zip(*matrix)]
    lower_price = max(row_minimums)
    upper_price = min(column_maximums)
    saddle_points = []
    if lower_price == upper_price:
        for row_index, minimum in enumerate(row_minimums):
            for column_index, maximum in enumerate(column_maximums):
                if minimum == lower_price and maximum == upper_price:
                    if matrix[row_index][column_index] == lower_price:
                        saddle_points.append((row_index + 1, column_index + 1))
    return row_minimums, column_maximums, lower_price, upper_price, saddle_points


def solve_two_by_n(matrix):
    """Solves a 2xn game by maximizing the lower envelope."""
    lines = [(matrix[0][column], matrix[1][column]) for column in range(len(matrix[0]))]
    candidates = candidate_probabilities(lines)
    values = {
        probability: min(line_value(start, end, probability) for start, end in lines)
        for probability in candidates
    }
    probability_a1 = max(values, key=lambda probability: values[probability])
    game_value = values[probability_a1]
    active_columns = [
        column
        for column, (start, end) in enumerate(lines)
        if line_value(start, end, probability_a1) == game_value
    ]

    strategy_b = [Fraction(0)] * len(lines)
    if len(active_columns) == 1:
        strategy_b[active_columns[0]] = Fraction(1)
    else:
        first, second = active_columns[:2]
        numerator = Fraction(matrix[1][second] - matrix[0][second])
        denominator = Fraction(
            matrix[0][first]
            - matrix[1][first]
            - matrix[0][second]
            + matrix[1][second]
        )
        probability_first = numerator / denominator
        strategy_b[first] = probability_first
        strategy_b[second] = 1 - probability_first

    return {
        "variable_probability": probability_a1,
        "game_value": game_value,
        "strategy_a": [probability_a1, 1 - probability_a1],
        "strategy_b": strategy_b,
        "active_lines": [column + 1 for column in active_columns],
        "envelope": "lower",
    }


def solve_m_by_two(matrix):
    """Solves an mx2 game by minimizing the upper envelope."""
    lines = [(row[0], row[1]) for row in matrix]
    candidates = candidate_probabilities(lines)
    values = {
        probability: max(line_value(start, end, probability) for start, end in lines)
        for probability in candidates
    }
    probability_b1 = min(values, key=lambda probability: values[probability])
    game_value = values[probability_b1]
    active_rows = [
        row
        for row, (start, end) in enumerate(lines)
        if line_value(start, end, probability_b1) == game_value
    ]

    strategy_a = [Fraction(0)] * len(lines)
    if len(active_rows) == 1:
        strategy_a[active_rows[0]] = Fraction(1)
    else:
        first, second = active_rows[:2]
        numerator = Fraction(matrix[second][1] - matrix[second][0])
        denominator = Fraction(
            matrix[first][0]
            - matrix[first][1]
            - matrix[second][0]
            + matrix[second][1]
        )
        probability_first = numerator / denominator
        strategy_a[first] = probability_first
        strategy_a[second] = 1 - probability_first

    return {
        "variable_probability": probability_b1,
        "game_value": game_value,
        "strategy_a": strategy_a,
        "strategy_b": [probability_b1, 1 - probability_b1],
        "active_lines": [row + 1 for row in active_rows],
        "envelope": "upper",
    }


def format_number(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_vector(values):
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def plot_solution(name, matrix, solution, output_path):
    """Plots all payoff lines, the relevant envelope, and the optimal point."""
    two_by_n = len(matrix) == 2
    lines = (
        [(matrix[0][column], matrix[1][column]) for column in range(len(matrix[0]))]
        if two_by_n
        else [(row[0], row[1]) for row in matrix]
    )
    labels = (
        [f"B{column + 1}" for column in range(len(lines))]
        if two_by_n
        else [f"A{row + 1}" for row in range(len(lines))]
    )
    probabilities = [index / 200 for index in range(201)]
    line_values = [
        [float(line_value(start, end, Fraction(index, 200))) for index in range(201)]
        for start, end in lines
    ]

    plt.figure(figsize=(8.5, 5.2))
    for values, label in zip(line_values, labels):
        plt.plot(probabilities, values, linewidth=1.25, label=label)

    if solution["envelope"] == "lower":
        envelope = [min(values[index] for values in line_values) for index in range(201)]
        envelope_label = "Lower envelope"
    else:
        envelope = [max(values[index] for values in line_values) for index in range(201)]
        envelope_label = "Upper envelope"
    plt.plot(probabilities, envelope, color="black", linewidth=3, label=envelope_label)

    optimum_x = float(solution["variable_probability"])
    optimum_y = float(solution["game_value"])
    plt.scatter([optimum_x], [optimum_y], color="red", zorder=5)
    plt.axvline(optimum_x, color="red", linestyle="--", linewidth=1)
    plt.axhline(optimum_y, color="red", linestyle="--", linewidth=1)
    plt.annotate(
        f"optimum ({format_number(solution['variable_probability'])}; "
        f"{format_number(solution['game_value'])})",
        (optimum_x, optimum_y),
        xytext=(8, 10),
        textcoords="offset points",
    )
    plt.xlabel("Probability of A1" if two_by_n else "Probability of B1")
    plt.ylabel("Expected payoff of player A")
    plt.title(name)
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)


def print_matrix(matrix):
    for row in matrix:
        print("  " + " ".join(f"{value:>4}" for value in row))


def main():
    output_directory = Path(__file__).resolve().parent

    for index, (name, matrix) in enumerate(MATRICES.items(), start=1):
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        print("Game matrix:")
        print_matrix(matrix)

        row_minimums, column_maximums, lower_price, upper_price, saddle_points = pure_prices(matrix)
        print("Row minimums:", row_minimums)
        print("Column maximums:", column_maximums)
        print("Lower price:", lower_price)
        print("Upper price:", upper_price)
        if saddle_points:
            print("Saddle points:", saddle_points)
        else:
            print("Saddle points: none")

        solution = solve_two_by_n(matrix) if len(matrix) == 2 else solve_m_by_two(matrix)
        print("Optimal strategy of player A:", format_vector(solution["strategy_a"]))
        print("Optimal strategy of player B:", format_vector(solution["strategy_b"]))
        print("Game value:", format_number(solution["game_value"]))
        print("Active graphical lines:", solution["active_lines"])

        file_name = f"matrix_{chr(64 + index)}.png"
        plot_solution(name, matrix, solution, output_directory / file_name)
        print("Plot saved to:", file_name)

    plt.show()


if __name__ == "__main__":
    main()
