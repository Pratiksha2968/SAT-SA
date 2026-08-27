"""src/analytics — supervisory analytics engine."""
from .rules import evaluate_rules, RULE_DEFINITIONS
from .negative_space import analyze_negative_space
from .process_mining import mine_processes
from .behavioural import build_behavioural_profiles
from .temporal import build_temporal_profiles
from .peer_analysis import analyze_peers
from .risk_engine import calculate_advanced_risk
from .root_cause import analyze_root_cause, risk_delta
from .what_if import simulate_what_if, what_if_grid
from .review_queue import build_review_queue

__all__ = [
    "evaluate_rules",
    "RULE_DEFINITIONS",
    "analyze_negative_space",
    "mine_processes",
    "build_behavioural_profiles",
    "build_temporal_profiles",
    "analyze_peers",
    "calculate_advanced_risk",
    "analyze_root_cause",
    "risk_delta",
    "simulate_what_if",
    "what_if_grid",
    "build_review_queue",
]
