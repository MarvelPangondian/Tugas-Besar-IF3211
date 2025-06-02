"""
Marine Ecosystem Simulation Package

This package provides a comprehensive agent-based model for simulating
marine ecosystem dynamics under various environmental conditions and
climate change scenarios.
"""

from .environment import Environment
from .agents import Agent, Phytoplankton, Zooplankton, Fish
from .simulation import MarineEcosystemSimulation
from .visualization import SimulationVisualizer

__version__ = "1.0.0"
__author__ = "Your Research Team"

__all__ = [
    'Environment',
    'Agent', 'Phytoplankton', 'Zooplankton', 'Fish',
    'MarineEcosystemSimulation',
    'SimulationVisualizer'
]