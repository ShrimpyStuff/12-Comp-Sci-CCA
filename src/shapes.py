import math

class BaseSphere():
    def __init__(self, r, thickness) -> None:
        self.r = r
        self.thickness = thickness

class Catenoid(BaseSphere):
    def __init__(self, r, thickness) -> None:
        super().__init__(r, thickness)


class Lattice(BaseSphere):
    def __init__(self, r, thickness, density) -> None:
        super().__init__(r, thickness)
        self.density = density