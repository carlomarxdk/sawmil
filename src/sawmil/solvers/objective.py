from __future__ import annotations
from dataclasses import dataclass
import logging

log = logging.getLogger("solvers.objective")


@dataclass
class Objective:
    """Container for quadratic/linear objective parts."""

    quadratic: float
    linear: float

    @property
    def objective(self) -> float:
        return self.quadratic + self.linear