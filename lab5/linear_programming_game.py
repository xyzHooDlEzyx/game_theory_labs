from fractions import Fraction


MATRIX = [
    [-2, 4, 0, 5],
    [3, -1, 4, 0],
    [-4, 0, -2, 1],
    [-3, -2, -1, 0],
    [-1, -3, 0, -1],
    [-2, -4, -1, -2],
]


def format_fraction(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_vector(vector):
    return "(" + "; ".join(format_fraction(value) for value in vector) + ")"


def print_matrix(matrix, title):
    print(title)
    for row in matrix:
        print("  " + " ".join(f"{format_fraction(value):>4}" for value in row))


def game_bounds(matrix):
    row_minimums = [min(row) for row in matrix]
    column_maximums = [max(row[column] for row in matrix) for column in range(len(matrix[0]))]
    return row_minimums, column_maximums, max(row_minimums), min(column_maximums)


def make_positive(matrix):
    smallest = min(min(row) for row in matrix)
    shift = 1 - smallest if smallest <= 0 else 0
    shifted = [[Fraction(value + shift) for value in row] for row in matrix]
    return shifted, shift


def pivot(tableau, pivot_row, pivot_column):
    pivot_value = tableau[pivot_row][pivot_column]
    tableau[pivot_row] = [value / pivot_value for value in tableau[pivot_row]]
    for row_index, row in enumerate(tableau):
        if row_index == pivot_row:
            continue
        multiplier = row[pivot_column]
        if multiplier:
            tableau[row_index] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(row, tableau[pivot_row])
            ]


def simplex_maximize(coefficients, constraints, right_hand_sides):
    """Solves max c*x subject to A*x <= b and x >= 0 by tableau simplex."""
    variable_count = len(coefficients)
    constraint_count = len(constraints)
    tableau = []
    basis = []

    for row_index, (row, right_hand_side) in enumerate(zip(constraints, right_hand_sides)):
        if right_hand_side < 0:
            raise ValueError("The initial simplex basis requires nonnegative right-hand sides.")
        slack = [Fraction(int(index == row_index)) for index in range(constraint_count)]
        tableau.append([Fraction(value) for value in row] + slack + [Fraction(right_hand_side)])
        basis.append(variable_count + row_index)

    tableau.append(
        [-Fraction(value) for value in coefficients]
        + [Fraction(0) for _ in range(constraint_count)]
        + [Fraction(0)]
    )

    iterations = []
    while True:
        objective = tableau[-1][:-1]
        entering = next((index for index, value in enumerate(objective) if value < 0), None)
        if entering is None:
            break

        candidates = []
        for row_index in range(constraint_count):
            coefficient = tableau[row_index][entering]
            if coefficient > 0:
                candidates.append((tableau[row_index][-1] / coefficient, row_index))
        if not candidates:
            raise ValueError("The linear program is unbounded.")

        _, leaving = min(candidates, key=lambda item: (item[0], basis[item[1]]))
        leaving_variable = basis[leaving]
        pivot(tableau, leaving, entering)
        basis[leaving] = entering
        iterations.append((entering, leaving_variable, tableau[-1][-1]))

    all_values = [Fraction(0) for _ in range(variable_count + constraint_count)]
    for row_index, basic_variable in enumerate(basis):
        all_values[basic_variable] = tableau[row_index][-1]

    primal = tuple(all_values[:variable_count])
    dual = tuple(tableau[-1][variable_count + index] for index in range(constraint_count))
    return primal, dual, tableau[-1][-1], iterations


def expected_payoffs(matrix, row_strategy, column_strategy):
    by_column = tuple(
        sum(Fraction(row_strategy[row]) * Fraction(matrix[row][column]) for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )
    by_row = tuple(
        sum(Fraction(matrix[row][column]) * Fraction(column_strategy[column]) for column in range(len(matrix[0])))
        for row in range(len(matrix))
    )
    total = sum(
        Fraction(row_strategy[row]) * Fraction(matrix[row][column]) * Fraction(column_strategy[column])
        for row in range(len(matrix))
        for column in range(len(matrix[0]))
    )
    return by_column, by_row, total


def solve_game(matrix):
    row_minimums, column_maximums, lower_value, upper_value = game_bounds(matrix)
    shifted, shift = make_positive(matrix)

    # The second player's transformed LP is max sum(y_j), A*y <= 1.
    columns = len(shifted[0])
    rows = len(shifted)
    y, x, objective, iterations = simplex_maximize(
        [Fraction(1) for _ in range(columns)],
        shifted,
        [Fraction(1) for _ in range(rows)],
    )
    if objective <= 0:
        raise ValueError("The transformed game value must be positive.")

    shifted_value = Fraction(1, 1) / objective
    row_strategy = tuple(value * shifted_value for value in x)
    column_strategy = tuple(value * shifted_value for value in y)
    original_value = shifted_value - shift
    by_column, by_row, expected_value = expected_payoffs(matrix, row_strategy, column_strategy)

    return {
        "row_minimums": row_minimums,
        "column_maximums": column_maximums,
        "lower_value": lower_value,
        "upper_value": upper_value,
        "shifted_matrix": shifted,
        "shift": shift,
        "x": x,
        "y": y,
        "objective": objective,
        "iterations": iterations,
        "shifted_value": shifted_value,
        "row_strategy": row_strategy,
        "column_strategy": column_strategy,
        "original_value": original_value,
        "payoffs_by_column": by_column,
        "payoffs_by_row": by_row,
        "expected_value": expected_value,
    }


def main():
    result = solve_game(MATRIX)
    print_matrix(MATRIX, "Original payoff matrix:")
    print("Row minimums:", result["row_minimums"])
    print("Column maximums:", result["column_maximums"])
    print("Lower game value:", result["lower_value"])
    print("Upper game value:", result["upper_value"])
    print("The game has a saddle point:", result["lower_value"] == result["upper_value"])

    print("\nPositive shift:", result["shift"])
    print_matrix(result["shifted_matrix"], "Shifted payoff matrix:")
    print("\nSimplex iterations:")
    for number, (entering, leaving, value) in enumerate(result["iterations"], start=1):
        print(
            f"  Iteration {number}: variable {entering + 1} enters, "
            f"variable {leaving + 1} leaves, objective = {format_fraction(value)}"
        )

    print("\nPlayer I transformed plan x:", format_vector(result["x"]))
    print("Player II transformed plan y:", format_vector(result["y"]))
    print("Dual objective values:", format_fraction(result["objective"]))
    print("Shifted game value:", format_fraction(result["shifted_value"]))
    print("Original game value:", format_fraction(result["original_value"]))
    print("Player I optimal strategy p:", format_vector(result["row_strategy"]))
    print("Player II optimal strategy q:", format_vector(result["column_strategy"]))
    print("Payoff against each pure column:", format_vector(result["payoffs_by_column"]))
    print("Payoff of each pure row:", format_vector(result["payoffs_by_row"]))
    print("Expected payoff:", format_fraction(result["expected_value"]))


if __name__ == "__main__":
    main()
