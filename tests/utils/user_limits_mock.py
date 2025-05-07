from aidial_rag.dial_user_limits import TokenStats, UserLimitsForModel


def user_limits_mock(minute_total=0, day_total=0, minute_used=0, day_used=0):
    assert minute_total >= minute_used
    assert day_total >= day_used

    async def _get_user_limits(_):
        return UserLimitsForModel(
            minuteTokenStats=TokenStats(used=minute_used, total=minute_total),
            dayTokenStats=TokenStats(used=day_used, total=day_total),
        )

    return _get_user_limits
