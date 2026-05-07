# src/__init__.py

# Expose core classes at the package level for cleaner imports
from .cv_pipeline import EmotionClassifier
from .state_manager import CognitiveBuffer
from .llm_orchestrator import MindSyncOrchestrator

__all__ = [
    "EmotionClassifier",
    "CognitiveBuffer",
    "MindSyncOrchestrator"
]