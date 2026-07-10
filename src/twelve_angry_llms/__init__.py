"""twelve-angry-llms: panel-based reliability annotation for preference data.

Run a panel of diverse LLM judges over (prompt, K responses) datapoints,
measure how much the judges agree on each one (inter-judge agreement, IJA),
and export TRL-ready (prompt, chosen, rejected) data with the reliability
signals attached.
"""

from .cache import CachedClient, ResponseCache
from .clients import LLMClient, OpenAICompatibleClient, OpenRouterClient
from .errors import JudgeError, ParseError, TwelveAngryError
from .export import to_hf_dataset, to_jsonl, to_records
from .judge import Judge
from .panel import Panel
from .protocols import RankingProtocol, ScoringProtocol
from .types import (
    DatapointResult,
    JudgeAnnotation,
    JudgeFailure,
    PanelDiagnostics,
    PreferenceDatapoint,
)

__version__ = "0.2.0"

__all__ = [
    "Panel",
    "Judge",
    "PreferenceDatapoint",
    "DatapointResult",
    "JudgeAnnotation",
    "JudgeFailure",
    "PanelDiagnostics",
    "ScoringProtocol",
    "RankingProtocol",
    "LLMClient",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "CachedClient",
    "ResponseCache",
    "to_records",
    "to_jsonl",
    "to_hf_dataset",
    "TwelveAngryError",
    "ParseError",
    "JudgeError",
]
