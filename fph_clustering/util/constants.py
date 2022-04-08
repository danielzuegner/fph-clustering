from enum import Enum


class Losses(Enum):
    TSD = 1
    DASGUPTA = 2

class ModelTypes(Enum):
    FPHDirectParameterization = 1
    FPHConstrainedDirectParameterization = 2