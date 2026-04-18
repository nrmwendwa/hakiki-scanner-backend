"""Deterministic decision engine that fuses pipeline signals into a final verdict."""

from .decision_engine import decide
from .schemas import DecisionResult, DecisionVerdict

__all__ = ["DecisionVerdict", "DecisionResult", "decide"]
