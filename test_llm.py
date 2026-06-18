"""
Step 1.2 - LLM 连接验证
测试 DeepSeek API 的连通性，包括普通响应和流式响应。
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def test_basic_response():
    """测试基础对话响应"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
        temperature=0.7,
    )
    messages = [HumanMessage(content="你好！请简单介绍一下你自己。")]
    response = llm.invoke(messages)
    print("=== 基础响应测试 ===")
    print(f"输入：你好！请简单介绍一下你自己。")
    print(f"输出：{response.content}")
    print(f"Token 用量：{response.usage_metadata}")
    print()
    return response


def test_streaming_response():
    """测试流式响应"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
        temperature=0.7,
        streaming=True,
    )
    messages = [HumanMessage(content="用一句话描述什么是向量数据库。")]
    print("=== 流式响应测试 ===")
    print("输入：用一句话描述什么是向量数据库。")
    print("输出：", end="", flush=True)
    full_response = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    print("\n")
    return full_response


if __name__ == "__main__":
    print("=== 开始测试 DeepSeek API 连接 ===\n")
    try:
        test_basic_response()
        test_streaming_response()
        print("全部测试通过！DeepSeek API 连接正常。")
    except Exception as e:
        print(f"测试失败：{e}")
        raise
