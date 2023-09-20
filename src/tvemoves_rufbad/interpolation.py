from .grid import Grid


class P1Interpolator:
    def __init__(self, grid: Grid, params: list):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(self, triangle: tuple[int, int, int], weights: tuple[float, float]):
        pass
