# coding:utf-8

from langchain.agents.middleware import AgentMiddleware


class LoggingMiddleware(AgentMiddleware):  # ✅ 类名随意
    """
    日志中间件 - 记录每次模型调用

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """

    def before_model(self, state, runtime):
        """模型调用前"""
        print("\n[中间件] before_model: 准备调用模型")
        print(f"[中间件] 当前消息数: {len(state.get('messages', []))}")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后"""
        print("[中间件] after_model: 模型已响应")
        last_message = state.get('messages', [])[-1]
        print(f"[中间件] 响应类型: {last_message.__class__.__name__}")
        return None  # 返回 None 表示不修改状态


class MessageTrimmerMiddleware(AgentMiddleware):
    """
    消息修剪中间件 - 限制消息数量

    before_model 修改消息列表
    注意：需要配合无 checkpointer 使用，否则历史会被恢复
    """

    def __init__(self, max_messages=5):
        super().__init__()
        self.max_messages = max_messages
        self.trimmed_count = 0  # 统计修剪次数

    def before_model(self, state, runtime):
        """模型调用前，修剪消息"""
        messages = state.get('messages', [])

        if len(messages) > self.max_messages:
            # 保留最近的 N 条消息
            trimmed_messages = messages[-self.max_messages:]
            self.trimmed_count += 1
            print(f"\n[修剪] 消息从 {len(messages)} 条减少到 {len(trimmed_messages)} 条 (第{self.trimmed_count}次修剪)")
            return {"messages": trimmed_messages}

        return None


class MaxCallsMiddleware(AgentMiddleware):
    """
    最大调用次数中间件 - 限制模型调用次数

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """

    def __init__(self, max_calls=3):
        super().__init__()
        self.max_calls = max_calls
        self.call_count = 0  # 统计调用次数

    def before_model(self, state, runtime):
        """模型调用前"""
        if self.call_count > self.max_calls:
            print("\n[中间件] 已达到最大调用次数，停止调用模型")
            raise ValueError(f"已达到最大调用次数限制: {self.max_calls}")
        return None

    def after_model(self, state, runtime):
        self.call_count += 1
        return None
