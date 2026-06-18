"""
Step 5.x - Agent 核心框架
集成记忆系统 + 情绪系统 + 多层级 Prompt 模板
"""
import os
from datetime import datetime
from typing import List, Optional, Generator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from memory_system import MemorySystem
from emotion_system import EmotionSystem

load_dotenv()


def load_system_prompt(path: str = "prompts/system_prompt.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class AgentCore:
    def __init__(self, model: str = None, temperature: float = 0.7, max_window_size: int = 10, auto_summarize_threshold: int = 15, memory_persist_dir: str = None, system_prompt_path: str = "prompts/system_prompt.txt"):
        self.llm = ChatOpenAI(
            model=model or os.getenv("MODEL_NAME", "deepseek-chat"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=temperature,
        )
        self.memory = MemorySystem(persist_dir=memory_persist_dir)
        self.emotion = EmotionSystem()
        self.history = ChatMessageHistory()
        self.max_window_size = max_window_size
        self.auto_summarize_threshold = auto_summarize_threshold
        self.conversation_round = 0
        self.system_template = load_system_prompt(system_prompt_path)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def chat(self, user_input: str) -> str:
        self.conversation_round += 1
        self.emotion.update_emotion(user_input)
        memories = self.memory.retrieve_relevant(user_input, n_results=5, threshold=0.3)
        context = self._build_context(user_input, memories)
        self.history.add_user_message(user_input)
        response = self.chain.invoke(context)
        self.history.add_ai_message(response)
        self._trim_history()
        if self.conversation_round % self.auto_summarize_threshold == 0:
            self._auto_summarize()
        self._save_interaction_to_memory(user_input, response)
        return response

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        self.conversation_round += 1
        self.emotion.update_emotion(user_input)
        memories = self.memory.retrieve_relevant(user_input, n_results=5, threshold=0.3)
        context = self._build_context(user_input, memories)
        self.history.add_user_message(user_input)
        full_response = ""
        for chunk in self.chain.stream(context):
            if chunk:
                full_response += chunk
                yield chunk
        self.history.add_ai_message(full_response)
        self._trim_history()
        if self.conversation_round % self.auto_summarize_threshold == 0:
            self._auto_summarize()
        self._save_interaction_to_memory(user_input, full_response)

    def _build_context(self, user_input: str, memories: List[dict]) -> dict:
        emotion_info = self.emotion.get_emotion_for_prompt()
        memory_text = self.memory.format_memories_for_prompt(memories, max_chars=600)
        return {"emotion": emotion_info, "memories": memory_text, "chat_history": self.history.messages, "input": user_input}

    def _trim_history(self):
        max_msgs = self.max_window_size * 2
        if len(self.history.messages) > max_msgs:
            self.history.messages = self.history.messages[-max_msgs:]

    def _save_interaction_to_memory(self, user_input: str, response: str):
        summary = f"用户说：{user_input[:100]}"
        self.memory.add_conversation_summary(summary=summary, emotion=self.emotion.current_emotion, importance=4)

    def _auto_summarize(self):
        recent = self.history.messages[-self.max_window_size * 2:]
        if len(recent) < 2:
            return
        dialogue_text = "\n".join([f"{'用户' if isinstance(m, HumanMessage) else '陈俊玮'}: {m.content[:150]}" for m in recent[-6:]])
        try:
            summary_msg = [HumanMessage(content=f"请用一句话总结以下对话的核心内容：\n\n{dialogue_text}")]
            summary = self.llm.invoke(summary_msg).content.strip()
            self.memory.add_conversation_summary(summary=summary, emotion=self.emotion.current_emotion, importance=6)
        except Exception:
            pass

    def get_status(self) -> dict:
        return {"emotion": self.emotion.current_emotion, "memory_count": self.memory.count(), "conversation_round": self.conversation_round, "history_size": len(self.history.messages)}

    def get_history(self) -> List[BaseMessage]:
        return self.history.messages

    def clear_history(self):
        self.history.clear()

    def reset(self):
        self.history.clear()
        self.emotion.reset()
        self.conversation_round = 0


def interactive_session():
    print("=== 陈俊玮 AI 陪伴对话 (完整版) ===\n")
    print("输入 'quit' 退出，'clear' 清空，'status' 查看状态\n")
    agent = AgentCore()
    while True:
        user_input = input("你 > ").strip()
        if user_input.lower() == "quit":
            print("\n小伴：好的，下次再聊~")
            break
        elif user_input.lower() == "clear":
            agent.reset()
            print("陈俊玮：已重置，我们重新开始吧~\n")
            continue
        elif user_input.lower() == "status":
            st = agent.get_status()
            print(f"[状态] 情绪:{st['emotion']} 记忆:{st['memory_count']}条 对话:{st['conversation_round']}轮\n")
            continue
        print("陈俊玮 > ", end="", flush=True)
        for chunk in agent.stream_chat(user_input):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    interactive_session()
