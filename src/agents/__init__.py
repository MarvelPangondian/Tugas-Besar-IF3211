"""
Agents Package

This package contains all the marine organism agents used in the ecosystem simulation.
"""

from .base_agent import Agent
from .phytoplankton import Phytoplankton
from .zooplankton import Zooplankton
from .fish import Fish

__all__ = ['Agent', 'Phytoplankton', 'Zooplankton', 'Fish']