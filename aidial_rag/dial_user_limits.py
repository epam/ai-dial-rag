import aiohttp
from pydantic import BaseModel, Field

from aidial_rag.dial_config import DialConfig


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
    dial_config: DialConfig, deployment_name: str
) -> UserLimitsForModel:
    """Returns the user limits for the specified model deployment.

    See https://epam-rail.com/dial_api#tag/Limits for the API documentation.
    """
    headers = {"Api-Key": dial_config.api_key.get_secret_value()}
    limits_url = (
        f"{dial_config.dial_url}/v1/deployments/{deployment_name}/limits"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(limits_url, headers=headers) as response:
            response.raise_for_status()
            limits_json = await response.json()
            return UserLimitsForModel.model_validate(limits_json)
