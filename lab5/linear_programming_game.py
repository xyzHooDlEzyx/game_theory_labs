from fractions import Fraction

import pulp


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


def as_fraction(value):
    """Converts a PuLP floating-point result to a compact rational value."""
    return Fraction(float(value)).limit_denominator(1_000_000)


def solve_linear_programs(matrix):
    """Solves both transformed game problems with PuLP's CBC solver."""
    rows = len(matrix)
    columns = len(matrix[0])
    solver = pulp.PULP_CBC_CMD(msg=False)

    player_i_problem = pulp.LpProblem("Player_I", pulp.LpMinimize)
    x_variables = [pulp.LpVariable(f"x_{row + 1}", lowBound=0) for row in range(rows)]
    player_i_problem += pulp.lpSum(x_variables)
    for column in range(columns):
        player_i_problem += (
            pulp.lpSum(matrix[row][column] * x_variables[row] for row in range(rows)) >= 1,
            f"column_{column + 1}",
        )

    player_ii_problem = pulp.LpProblem("Player_II", pulp.LpMaximize)
    y_variables = [
        pulp.LpVariable(f"y_{column + 1}", lowBound=0) for column in range(columns)
    ]
    player_ii_problem += pulp.lpSum(y_variables)
    for row in range(rows):
        player_ii_problem += (
            pulp.lpSum(matrix[row][column] * y_variables[column] for column in range(columns)) <= 1,
            f"row_{row + 1}",
        )

    player_i_problem.solve(solver)
    player_ii_problem.solve(solver)
    player_i_status = pulp.LpStatus[player_i_problem.status]
    player_ii_status = pulp.LpStatus[player_ii_problem.status]
    if player_i_status != "Optimal" or player_ii_status != "Optimal":
        raise ValueError(
            "PuLP did not find optimal solutions: "
            f"Player I = {player_i_status}, Player II = {player_ii_status}."
        )

    x = tuple(as_fraction(variable.value()) for variable in x_variables)
    y = tuple(as_fraction(variable.value()) for variable in y_variables)
    player_i_objective = as_fraction(pulp.value(player_i_problem.objective))
    player_ii_objective = as_fraction(pulp.value(player_ii_problem.objective))
    return x, y, player_i_objective, player_ii_objective, player_i_status, player_ii_status


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

    x, y, player_i_objective, player_ii_objective, player_i_status, player_ii_status = (
        solve_linear_programs(shifted)
    )
    if player_i_objective != player_ii_objective:
        raise ValueError("The primal and dual objective values are different.")
    if player_i_objective <= 0:
        raise ValueError("The transformed game value must be positive.")

    shifted_value = Fraction(1, 1) / player_i_objective
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
        "player_i_objective": player_i_objective,
        "player_ii_objective": player_ii_objective,
        "player_i_status": player_i_status,
        "player_ii_status": player_ii_status,
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
    print("\nPuLP solver status for Player I:", result["player_i_status"])
    print("PuLP solver status for Player II:", result["player_ii_status"])
    print("Player I transformed plan x:", format_vector(result["x"]))
    print("Player II transformed plan y:", format_vector(result["y"]))
    print("Player I objective value:", format_fraction(result["player_i_objective"]))
    print("Player II objective value:", format_fraction(result["player_ii_objective"]))
    print("Shifted game value:", format_fraction(result["shifted_value"]))
    print("Original game value:", format_fraction(result["original_value"]))
    print("Player I optimal strategy p:", format_vector(result["row_strategy"]))
    print("Player II optimal strategy q:", format_vector(result["column_strategy"]))
    print("Payoff against each pure column:", format_vector(result["payoffs_by_column"]))
    print("Payoff of each pure row:", format_vector(result["payoffs_by_row"]))
    print("Expected payoff:", format_fraction(result["expected_value"]))


if __name__ == "__main__":
    main()
