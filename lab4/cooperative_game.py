from fractions import Fraction
from itertools import combinations
from math import factorial


PLAYERS = frozenset({1, 2, 3})

VALUES = {
    frozenset(): 0,
    frozenset({1}): 900,
    frozenset({2}): 850,
    frozenset({3}): 1200,
    frozenset({1, 2}): 2000,
    frozenset({1, 3}): 2400,
    frozenset({2, 3}): 2500,
    frozenset({1, 2, 3}): 3600,
}


def coalitions(players=PLAYERS):
    """Returns all coalitions in increasing order of size."""
    ordered_players = sorted(players)
    result = []
    for size in range(len(ordered_players) + 1):
        result.extend(frozenset(group) for group in combinations(ordered_players, size))
    return result


def coalition_name(coalition):
    if not coalition:
        return "empty coalition"
    return "{" + ", ".join(str(player) for player in sorted(coalition)) + "}"


def format_number(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_vector(vector):
    return "(" + "; ".join(format_number(value) for value in vector) + ")"


def check_superadditivity(values, players=PLAYERS):
    """Checks V(A union B) >= V(A) + V(B) for all disjoint coalitions."""
    checks = []
    all_coalitions = coalitions(players)
    for index, first in enumerate(all_coalitions):
        for second in all_coalitions[index + 1 :]:
            if first and second and first.isdisjoint(second):
                left = Fraction(values[first | second])
                right = Fraction(values[first]) + Fraction(values[second])
                checks.append((first, second, left, right, left >= right))
    return checks


def normalize_game(values, players=PLAYERS):
    """Returns the 0-1 normalization of an essential cooperative game."""
    individual_total = sum(Fraction(values[frozenset({player})]) for player in players)
    surplus = Fraction(values[players]) - individual_total
    if surplus <= 0:
        raise ValueError("The game must be essential for 0-1 normalization.")

    normalized = {}
    for coalition in coalitions(players):
        individual_value = sum(
            Fraction(values[frozenset({player})]) for player in coalition
        )
        normalized[coalition] = (Fraction(values[coalition]) - individual_value) / surplus
    return normalized, surplus


def solve_linear_system(matrix, vector):
    """Solves a square linear system exactly by Gaussian elimination."""
    size = len(vector)
    augmented = [
        [Fraction(value) for value in matrix[row]] + [Fraction(vector[row])]
        for row in range(size)
    ]

    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]

        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                augmented[row][item] - multiplier * augmented[column][item]
                for item in range(size + 1)
            ]
    return tuple(augmented[row][-1] for row in range(size))


def belongs_to_core(allocation, values, players=PLAYERS):
    """Checks efficiency and every coalition-rationality inequality."""
    ordered_players = sorted(players)
    payments = dict(zip(ordered_players, map(Fraction, allocation)))
    if sum(payments.values()) != Fraction(values[players]):
        return False
    return all(
        sum(payments[player] for player in coalition) >= Fraction(values[coalition])
        for coalition in coalitions(players)
    )


def core_vertices(values, players=PLAYERS):
    """Finds all vertices of a three-player core by intersecting active constraints."""
    ordered_players = sorted(players)
    proper_coalitions = [
        coalition
        for coalition in coalitions(players)
        if coalition and coalition != players
    ]
    efficiency_row = [Fraction(1)] * len(ordered_players)
    efficiency_value = Fraction(values[players])
    vertices = set()

    for first, second in combinations(proper_coalitions, 2):
        matrix = [
            efficiency_row,
            [Fraction(player in first) for player in ordered_players],
            [Fraction(player in second) for player in ordered_players],
        ]
        vector = [efficiency_value, Fraction(values[first]), Fraction(values[second])]
        solution = solve_linear_system(matrix, vector)
        if solution is not None and belongs_to_core(solution, values, players):
            vertices.add(solution)
    return sorted(vertices)


def shapley_value(values, players=PLAYERS):
    """Computes the Shapley value from exact marginal contributions."""
    ordered_players = sorted(players)
    player_count = len(ordered_players)
    result = []
    for player in ordered_players:
        others = players - {player}
        component = Fraction(0)
        for coalition in coalitions(others):
            size = len(coalition)
            weight = Fraction(
                factorial(size) * factorial(player_count - size - 1),
                factorial(player_count),
            )
            marginal = Fraction(values[coalition | {player}]) - Fraction(values[coalition])
            component += weight * marginal
        result.append(component)
    return tuple(result)


def print_characteristic_function(values):
    print("Characteristic function:")
    for coalition in coalitions():
        print(f"  V({coalition_name(coalition)}) = {format_number(values[coalition])}")


def main():
    print_characteristic_function(VALUES)

    print("\nSuperadditivity checks:")
    checks = check_superadditivity(VALUES)
    for first, second, union_value, separate_value, passed in checks:
        status = "passed" if passed else "failed"
        print(
            f"  {coalition_name(first)} + {coalition_name(second)}: "
            f"{format_number(union_value)} >= {format_number(separate_value)} - {status}"
        )
    print("The game is superadditive:", all(item[-1] for item in checks))

    individual_total = sum(VALUES[frozenset({player})] for player in PLAYERS)
    grand_value = VALUES[PLAYERS]
    print("\nSum of individual values:", individual_total)
    print("Grand coalition value:", grand_value)
    print("The game is essential:", individual_total < grand_value)

    normalized, surplus = normalize_game(VALUES)
    print("\nCooperation surplus:", format_number(surplus))
    print("Normalized characteristic function:")
    for coalition in coalitions():
        print(f"  V'({coalition_name(coalition)}) = {format_number(normalized[coalition])}")

    print("\nSufficient nonempty-core condition from Theorem 9:")
    theorem_checks = []
    for coalition in coalitions():
        if not coalition:
            continue
        bound = Fraction(1, len(PLAYERS) - len(coalition) + 1)
        passed = normalized[coalition] <= bound
        theorem_checks.append(passed)
        print(
            f"  {coalition_name(coalition)}: {format_number(normalized[coalition])} "
            f"<= {format_number(bound)} - {'passed' if passed else 'failed'}"
        )
    print("The sufficient condition is satisfied:", all(theorem_checks))

    vertices = core_vertices(VALUES)
    print("\nCore vertices:")
    for vertex in vertices:
        print(" ", format_vector(vertex))
    selected_allocation = (Fraction(1050), Fraction(1050), Fraction(1500))
    print("Selected core allocation:", format_vector(selected_allocation))
    print("Selected allocation belongs to the core:", belongs_to_core(selected_allocation, VALUES))

    shapley = shapley_value(VALUES)
    print("\nShapley vector:", format_vector(shapley))
    print("Shapley vector total:", format_number(sum(shapley)))
    print("Shapley vector belongs to the core:", belongs_to_core(shapley, VALUES))


if __name__ == "__main__":
    main()
