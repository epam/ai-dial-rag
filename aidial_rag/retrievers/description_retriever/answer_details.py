"""
Answer detail level enum
"""

import enum


@enum.unique
class AnswerDetails(enum.Enum):
    SHORT = "Short"
    DETAILED = "Detailed"
    VERY_DETAILED = "Very detailed"
