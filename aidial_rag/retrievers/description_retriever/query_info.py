"""
Information about LLM query
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class QueryInfo:
    """
    Information about LLM query
    """

    prompt_tokens: int
    completion_tokens: int
    total_cost: float
    information: list[str]
    error_code: int = 0

    @classmethod
    def Empty(cls) -> "QueryInfo":
        return QueryInfo(
            prompt_tokens=0, completion_tokens=0, total_cost=0.0, information=[]
        )

    @classmethod
    def Error(cls, error_message: str) -> "QueryInfo":
        return QueryInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_cost=0.0,
            information=[error_message],
            error_code=-1,
        )

    @classmethod
    def ErrorWithCode(cls, error_code: int, error_message: str) -> "QueryInfo":
        return QueryInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_cost=0.0,
            information=[error_message],
            error_code=error_code,
        )

    def update_tokens(self, cb):
        """
        Update query info
        """
        self.prompt_tokens = cb.prompt_tokens
        self.completion_tokens = cb.completion_tokens
        self.total_cost = cb.total_cost
