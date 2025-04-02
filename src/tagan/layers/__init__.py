"""
Neural network layers for the TAGAN architecture.
"""

from .geometric_attention import GeometricAttention
from .temporal_propagation import TemporalPropagation
from .temporal_attention import TemporalAttention
from .classification import ClassificationModule
from .graph_attention import TAGANGraphAttention