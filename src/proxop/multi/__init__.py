"""
Compute the proximity operator and the evaluation of multivariate functions.

Author: Mbaye DIONGUE (Jun 2022)
"""

from .AbsDiff import *
from .AffineBarrier import *
from .ChiSquare import *
from .Hellinger import *
from .HyperslabBarrier import *
from .Ialpha import *
from .InvPenaltyLogDet import *
from .Jeffrey import *
from .Kullback import *
from .L2BallBarrier import *
from .L2DownConv import *
from .L2Norm import *
from .L2Positive import *
from .L21columns import *
from .L21rows import *
from .Linf import *
from .Linf1 import *
from .LogDet import *
from .Max import *
from .NeumannEntropy import *
from .NuclearBall import *
from .NuclearBlocks import *
from .NuclearLogDet import *
from .NuclearNorm import *
from .NuclearNormRidge import *
from .PermutationInvariant import *
from .PersAbsDiff import *
from .PersHuber import *
from .PersHuber import *
from .PersSqrt import *
from .PersSquare import *
from .PersVapnik import *
from .Rank import *
from .RankRidge import *
from .Renyi import *
from .Schatten3_2Penalty import *
from .Schatten3Penalty import *
from .Schatten4_3Penalty import *
from .Schatten4Penalty import *
from .SchattenPenaltyLogDet import *
from .Spectral import *
from .SquaredFrobeniusNormLogDet import *
from .Vapnik import *


__all__ = [
    "AbsDiff",
    "AffineBarrier",
    "ChiSquare",
    "Hellinger",
    "HyperslabBarrier",
    "Ialpha",
    "InvPenaltyLogDet",
    "Jeffrey",
    "Kullback",
    "L2BallBarrier",
    "L2DownConv",
    "L2Norm",
    "L2Positive",
    "L21columns",
    "L21rows",
    "Linf",
    "Linf1",
    "LogDet",
    "Max",
    "NeumannEntropy",
    "NuclearBall",
    "NuclearBlocks",
    "NuclearLogDet",
    "NuclearNorm",
    "NuclearNormRidge",
    "PermutationInvariant",
    "PersAbsDiff",
    "PersHuber",
    "PersHuber",
    "PersSqrt",
    "PersSquare",
    "PersVapnik",
    "Rank",
    "RankRidge",
    "Renyi",
    "Schatten3_2Penalty",
    "Schatten3Penalty",
    "Schatten4_3Penalty",
    "Schatten4Penalty",
    "SchattenPenaltyLogDet",
    "Spectral",
    "SquaredFrobeniusNormLogDet",
    "Vapnik",
]
