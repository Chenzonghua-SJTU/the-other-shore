"""Opt-in prompt overlay for the optional tool, independent of filter numerics."""

VERSION = 'filter-guidance-v1'
MARKER = '联合滤波可选工具使用说明（'+VERSION+'）'
OLD_DESCRIPTION = '仅依据公开噪声尺度做局部多项式平滑。'
NEW_DESCRIPTION = '默认预处理依据公开噪声尺度做局部多项式平滑，也可选择联合位置—速度样条。'
GUIDANCE = ('\n\n'+MARKER+'：'
    '已有至少一条含噪轨迹达到12个观测点，且预算允许完成后续搜索、拟合、验证和提交时，优先尝试'
    'denoise_trajectories(train_probe_ids=所选轨迹,time_budget_seconds=2)，'
    '再将返回的preprocessing_id连同相同的train_probe_ids传给symbolic_regression。'
    '这是可选的搜索预处理，不保证加速度或规律更准确，不要求每次搜索都使用；'
    '短轨迹、零噪声、预算不足或滤波失败时使用旧预处理，调用symbolic_regression时省略preprocessing_id。'
    '不要重复平滑派生数据；最终候选拟合、rollout与验证继续依据原始观测。')


class FilterGuidanceClient:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    def complete(self, messages, **kwargs):
        # Modify the actual request messages, so the original model client's
        # full model_messages.jsonl logging captures exactly what the model sees.
        if messages and messages[0].get('role') == 'system':
            content = messages[0].get('content')
            if isinstance(content, str) and MARKER not in content:
                messages[0]['content'] = content.replace(OLD_DESCRIPTION, NEW_DESCRIPTION)+GUIDANCE
        return self._client.complete(messages, **kwargs)


def configure(client, *, enabled=False):
    if type(enabled) is not bool:
        raise ValueError('Filter guidance enabled must be boolean')
    return FilterGuidanceClient(client) if enabled else client
