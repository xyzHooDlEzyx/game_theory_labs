A = [[-4, -3],
    [-2, 5],
    [3, 4],
    [1, 2],
    [0, 1],
    [-1, 3]]

B = [[-3,-2],
    [-4,5],
    [2,4],
    [2,3],
    [0,1],
    [1,5]]

C = [[-4,5],
    [4,-3],
    [-1,-2],
    [0,-1],
    [2,1],
    [3,-4]]

D = [[-4,2],
    [-1,5],
    [1,4],
    [2,3],
    [0,1],
    [-2,0]]

E = [[-4,-2],
    [-3,5],
    [3,4],
    [3,5],
    [0,1],
    [1,2]]

F = [[-4,5],
    [5,-4],
    [-2,-1],
    [0,-2],
    [2,1],
    [3,-3]]

class Game:
    def __init__(self, matrix):
        self.matrix = matrix
        self._row = [min(row) for row in self.matrix]
        self._column = [max(column) for column in zip(*self.matrix)]
        self.can_be_solved = False

    def _find_max_min(self):
        lower_price = max(self._row)    
        strategies = tuple(
            index+1
            for index, value in enumerate(self._row)
            if value == lower_price
        )

        return lower_price, strategies
    
    def _find_min_max(self):
        upper_price = min(self._column)
        strategies = tuple(
            index+1
            for index, value in enumerate(self._column)
            if value == upper_price
        )
        return upper_price, strategies
    
    def _find_saddle_points(self, strategies_a, strategies_b):
        saddle_points=[]
        
        for i in strategies_a:
            for j in strategies_b:
                value = self.matrix[i - 1][j - 1]
                saddle_points.append((i, j, f"with value: {value}"))

        return tuple(saddle_points)
        


    def solve_matrix(self):
        lower_price, strategies_a = self._find_max_min()
        upper_price, strategies_b = self._find_min_max()
        self.can_be_solved = lower_price == upper_price

        saddle_points = ()
        if self.can_be_solved:
            saddle_points = self._find_saddle_points(strategies_a, strategies_b)

        self._print_game(
            lower_price,
            upper_price,
            strategies_a,
            strategies_b,
            saddle_points,
            self.can_be_solved,
        )

        return {
            "lower_price": lower_price,
            "upper_price": upper_price,
            "strategies_a": strategies_a,
            "strategies_b": strategies_b,
            "saddle_points": saddle_points,
            "can_be_solved": self.can_be_solved,
        }
        
    def _print_game(self, lower_price, upper_price, strategies_a, strategies_b, saddle_points, can_be_solved):
        print("Game matrix:")
        for row in self.matrix:
            print("  " + " ".join(f"{value:>3}" for value in row))

        print(f"Row minimums: {self._row}")
        print(f"Lower price (maximin): {lower_price}")
        print(f"Player A optimal strategies: {strategies_a}")
        print(f"Column maximums: {self._column}")
        print(f"Upper price (minimax): {upper_price}")
        print(f"Player B optimal strategies: {strategies_b}")

        if can_be_solved:
            print(f"The game has a pure-strategy solution with value {lower_price}.")
            print(f"Saddle points: {saddle_points}")
        else:
            print("The game has no pure-strategy solution or saddle points.")





if __name__ == "__main__":
    examples = (
        ("One saddle point", (("A", A), ("D", D))),
        ("Multiple saddle points", (("B", B), ("E", E))),
        ("No saddle points", (("C", C), ("F", F))),
    )

    for category, matrices in examples:
        print(f"\n{'=' * 60}\n{category}\n{'=' * 60}")
        for name, matrix in matrices:
            print(f"\nMatrix {name}")
            Game(matrix).solve_matrix()
    
