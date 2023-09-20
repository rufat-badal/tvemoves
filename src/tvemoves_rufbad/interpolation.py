class P1Interpolator:
    def __init__(self, grid, params):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(self, triangle, weights):
        pass
