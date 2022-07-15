"""
Compute the proximity operator and the evaluation of 'scalar' functions.

Note: When the input is a vector, the proximity operator is computed element-wise
(i.e. component by component) thanks to the property of proximity operator.

Author: Mbaye DIONGUE (Jun 2022)
"""


from .AbsValue import *
from .ArctanActi import *
from .ArgsinhActi import *
from .Bathtub import *
from .BentIdentity import *
from .Berhu import *
from .BregAbsEntropy import *
from .BregAbsLog import *
from .BregBoltzShannon import *
from .BregBoltzShannon2 import *
from .BregExp import *
from .BregLpNorm import *
from .BregSquareLog import *
from .ElliotActi import *
from .ELUacti import *
from .Entropy import *
from .Exp import *
from .FairPotential import *
from .FermiDiracEntropy import *
from .GMActi import *
from .HingeLoss import *
from .Huber import *
from .Hyperbolic import *
from .Inverse import *
from .ISRLU import *
from .ISRU import *
from .Log import *
from .LogActi import *
from .LogBarrier import *
from .LogInverse import *
from .LogisticLoss import *
from .LogPower import *
from .NegativeRoot import *
from .Power import *
from .PRelu import *
from .SmoothedFermiDiracEntropy import *
from .Square import *
from .SquaredBathtub import *
from .SquaredHinge import *
from .TanhActi import *
from .Thresholder import *
from .UnimodalSigmoid import *

__all__ = [
    "AbsValue",
    "ArctanActi",
    "ArgsinhActi",
    "Bathtub",
    "BentIdentity",
    "Berhu",
    "BregAbsEntropy",
    "BregAbsLog",
    "BregBoltzShannon",
    "BregBoltzShannon2",
    "BregExp",
    "BregLpNorm",
    "BregSquareLog",
    "ElliotActi",
    "ELUacti",
    "Entropy",
    "Exp",
    "FairPotential",
    "FermiDiracEntropy",
    "GMActi",
    "HingeLoss",
    "Huber",
    "Hyperbolic",
    "Inverse",
    "ISRLU",
    "ISRU",
    "Log",
    "LogActi",
    "LogBarrier",
    "LogInverse",
    "LogisticLoss",
    "LogPower",
    "NegativeRoot",
    "Power",
    "PRelu",
    "SmoothedFermiDiracEntropy",
    "Square",
    "SquaredBathtub",
    "SquaredHinge",
    "TanhActi",
    "Thresholder",
    "UnimodalSigmoid",
]
