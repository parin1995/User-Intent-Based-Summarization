__all__ = [
    "LooperException",
    "StopLoopingException",
    "EarlyStoppingException",
]


class LooperException(Exception):
    """Base class for exceptions encountered during looping"""

    pass


class StopLoopingException(LooperException):
    """Base class for exceptions which should stop training in this module."""

    pass


class EarlyStoppingException(StopLoopingException):
    """Max Value Exceeded"""
    def __init__(self,
                 condition: str):
        self.condition = condition

    def __str__(self):
        return f"EarlyStopping: {self.condition}"