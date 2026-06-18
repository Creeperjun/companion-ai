import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

DEFAULT_EMOTION = "平静"


def load_system_prompt(path: str = "prompts/system_prompt.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class CompanionChat:
    def __init__(
        self,
        system_prompt_path: str = "prompts/system_prompt.txt",
        model: str = None,
        temperature: float = 0.7,
        max_window_size: int = 10,
    ):
        self.llm = ChatOpenAI(
            model=model or os.getenv("MODEL_NAME", "deepseek-chat"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=temperature,
        )
        self.system_template = load_system_prompt(system_prompt_path)
        self.max_window_size = max_window_size
        self.history = ChatMessageHistory()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def build_context(self, user_input: str, emotion: str = DEFAULT_EMOTION, memories: str = "暂无相关记忆。") -> dict:
        return {
            "emotion": emotion,
            "memories": memories,
            "chat_history": self.history.messages,
            "input": user_input,
        }

    def chat(self, user_input: str, emotion: str = DEFAULT_EMOTION, memories: str = "暂无相关记忆。") -> str:
        self.history.add_user_message(user_input)
        context = self.build_context(user_input, emotion, memories)
        response = self.chain.invoke(context)
        self.history.add_ai_message(response)
        self._trim_history()
        return response

    def stream_chat(self, user_input: str, emotion: str = DEFAULT_EMOTION, memories: str = "暂无相关记忆。"):
        self.history.add_user_message(user_input)
        context = self.build_context(user_input, emotion, memories)
        full_response = ""
        for chunk in self.chain.stream(context):
            if chunk:
                full_response += chunk
                yield chunk
        self.history.add_ai_message(full_response)
        self._trim_history()

    def _trim_history(self):
        max_messages = self.max_window_size * 2
        if len(self.history.messages) > max_messages:
            self.history.messages = self.history.messages[-max_messages:]

    def get_history(self) -> List[BaseMessage]:
        return self.history.messages

    def clear_history(self):
        self.history.clear()


def interactive_session():
    print("=== 陈俊玮 AI 陪伴对话 ===\n")
    print("输入 'quit' 退出，输入 'clear' 清空历史\n")
    companion = CompanionChat()
    while True:
        user_input = input("你 > ").strip()
        if user_input.lower() == "quit":
            print("\n小伴：好的，下次再聊~")
            break
        elif user_input.lower() == "clear":
            companion.clear_history()
            print("陈俊玮：已清空对话历史，我们重新开始吧~\n")
            continue
        print("陈俊玮 > ", end="", flush=True)
        for chunk in companion.stream_chat(user_input):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    interactive_session()
