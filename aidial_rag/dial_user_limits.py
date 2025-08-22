from pydantic import BaseModel, Field

from aidial_rag.dial_api_client import DialApiClient


class TokenStats(BaseModel):
    total: int
    used: int


class UserLimitsForModel(BaseModel):
    """Implementation of the response from the /v1/deployments/{deployment_name}/limits endpoint

    See https://epam-rail.com/dial_api#tag/Limits for the API documentation.
    """

    minute_token_stats: TokenStats = Field(alias="minuteTokenStats")
    day_token_stats: TokenStats = Field(alias="dayTokenStats")


async def get_user_limits_for_model(
    dial_api_client: DialApiClient, deployment_name: str
) -> UserLimitsForModel:
    """Returns the user limits for the specified model deployment.

    See https://epam-rail.com/dial_api#tag/Limits for the API documentation.
    """

    limits_relative_url = f"deployments/{deployment_name}/limits"
    async with dial_api_client.session.get(limits_relative_url) as response:
        response.raise_for_status()
        limits_json = await response.json()
        return UserLimitsForModel.model_validate(limits_json)
