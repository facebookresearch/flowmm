"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from flowmm.rfm.manifolds.analog_bits import MultiAtomAnalogBits
from flowmm.rfm.manifolds.euclidean import EuclideanWithLogProb
from flowmm.rfm.manifolds.flat_torus import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
)
from flowmm.rfm.manifolds.null import NullManifoldWithDeltaRandom
from flowmm.rfm.manifolds.product import ProductManifoldWithLogProb
from flowmm.rfm.manifolds.simplex import (
    FlatDirichletSimplex,
    MultiAtomFlatDirichletSimplex,
)
