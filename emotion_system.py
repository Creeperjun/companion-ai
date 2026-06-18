"""
Step 4.x — 情绪状态机
定义情绪状态、状态转换、情绪影响对话逻辑
"""
import os
import json
from datetime import datetime
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ===== 情绪状态定义 =====

EMOTION_STATES = ["愉悦", "平静", "悲伤", "惊讶", "生气"]

# 情绪转换规则矩阵: 从某情绪 -> 可转换到的情绪列表
EMOTION_TRANSITIONS = {
    "愉悦": ["愉悦", "平静", "惊讶"],
    "平静": ["愉悦", "平静", "悲伤", "惊讶", "生气"],
    "悲伤": ["悲伤", "平静", "惊讶"],
    "惊讶": ["愉悦", "平静", "惊讶", "生气"],
    "生气": ["生气", "平静", "惊讶"],
}

# 各情绪对应的回复风格提示
EMOTION_STYLE_PROMPTS = {
    "愉悦": "你此刻心情愉悦，语气轻快活泼，多用感叹和温暖的表达。",
    "平静": "你此刻心情平静温和，语气温柔而沉稳，给人安心的感觉。",
    "悲伤": "你此刻感到有些悲伤，语气柔和低沉，带着共情和陪伴感，但不要过度消极。",
    "惊讶": "你此刻感到惊讶，语气带着好奇和兴奋，表达出意外和关注。",
    "生气": "你此刻有些生气，语气略显不满但仍保持克制，表达理解和立场但不失礼貌。",
}


class EmotionSystem:
    """情绪状态机"""

    def __init__(self, initial_emotion: str = "平静", model: str = None, temperature: float = 0.3):
        if initial_emotion not in EMOTION_STATES:
            initial_emotion = "平静"
        self.current_emotion = initial_emotion
        self.emotion_history: List[dict] = []

        # 用于情绪分析的 LLM（低温度，确保输出结构化）
        self.analyzer_llm = ChatOpenAI(
            model=model or os.getenv("EMOTION_MODEL", os.getenv("MODEL_NAME", "deepseek-chat")),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=temperature,
        )
        self.parser = StrOutputParser()

        # 情绪分析 Prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "你是一个情绪分析专家。分析用户输入的文本，判断其中蕴含的情绪。\n\n"
                "可选的输出情感（仅输出一个词）：愉悦 / 平静 / 悲伤 / 惊讶 / 生气\n\n"
                "判断规则：\n"
                "- 愉悦：开心、兴奋、满足、感激、喜爱\n"
                "- 平静：中性、平淡、正常、无特别情绪\n"
                "- 悲伤：失落、难过、孤独、沮丧、失望\n"
                "- 惊讶：意外、震惊、惊喜、困惑\n"
                "- 生气：愤怒、不满、烦躁、抱怨、不耐烦\n\n"
                "仅输出一个情感词，不要输出其他内容。"
            )),
            ("human", "{user_input}"),
        ])
        self.analysis_chain = self.analysis_prompt | self.analyzer_llm | self.parser

    def analyze_emotion(self, user_input: str) -> str:
        """分析用户输入的情绪"""
        try:
            result = self.analysis_chain.invoke({"user_input": user_input}).strip()
            # 清理可能的标点或空格
            result = result.replace("。", "").replace("！", "").replace("？", "").strip()
            if result in EMOTION_STATES:
                return result
        except Exception:
            pass
        return "平静"

    def _apply_transition_rules(self, user_emotion: str) -> str:
        """根据转换规则和当前情绪，决定新的情绪"""
        allowed = EMOTION_TRANSITIONS.get(self.current_emotion, EMOTION_STATES)

        if user_emotion in allowed:
            return user_emotion

        # 如果不允许直接跳转，找一个最近的允许状态
        for state in ["平静", "惊讶", "愉悦", "悲伤", "生气"]:
            if state in allowed:
                return state

        return "平静"

    def update_emotion(self, user_input: str) -> str:
        """根据用户输入更新当前情绪"""
        detected = self.analyze_emotion(user_input)
        new_emotion = self._apply_transition_rules(detected)

        self.emotion_history.append({
            "timestamp": datetime.now().isoformat(),
            "from": self.current_emotion,
            "to": new_emotion,
            "detected_user_emotion": detected,
            "trigger": user_input[:100],
        })

        self.current_emotion = new_emotion
        return self.current_emotion

    def get_emotion_style_prompt(self) -> str:
        """获取当前情绪对应的风格提示"""
        return EMOTION_STYLE_PROMPTS.get(self.current_emotion, EMOTION_STYLE_PROMPTS["平静"])

    def get_emotion_info(self) -> dict:
        """获取当前情绪状态摘要"""
        return {
            "current": self.current_emotion,
            "style_prompt": self.get_emotion_style_prompt(),
            "history": self.emotion_history[-5:] if self.emotion_history else [],
        }

    def get_emotion_for_prompt(self) -> str:
        """获取注入 Prompt 的情绪描述"""
        return f"当前情绪：{self.current_emotion}。{self.get_emotion_style_prompt()}"

    def reset(self, emotion: str = "平静"):
        """重置情绪状态"""
        if emotion in EMOTION_STATES:
            self.current_emotion = emotion


# ===== 情感分析结果缓存（可选） =====

def classify_user_sentiment_direct(user_input: str, llm) -> Tuple[str, float]:
    """直接基于关键词的快速情绪分类（备选方案，不依赖 LLM）"""
    keywords = {
        "愉悦": ["开心", "高兴", "太好了", "喜欢", "爱", "棒", "快乐", "幸福", "感谢", "谢谢", "哈哈"],
        "悲伤": ["难过", "伤心", "失落", "孤独", "难受", "郁闷", "不开心", "哭", "悲伤", "沮丧"],
        "生气": ["生气", "烦", "讨厌", "可恶", "气死", "受不了", "愤怒", "烦躁"],
        "惊讶": ["哇", "真的吗", "不敢相信", "天哪", "居然", "竟然", "没想到", "惊讶"],
    }
    scores = {e: 0 for e in EMOTION_STATES}
    for emotion, words in keywords.items():
        for word in words:
            if word in user_input:
                scores[emotion] += 1

    # 如果有关键词匹配，返回得分最高的
    max_score = max(scores.values())
    if max_score > 0:
        best = [e for e, s in scores.items() if s == max_score][0]
        return best, max_score / max(len(user_input), 1)
    return "平静", 0.0


if __name__ == "__main__":
    # 测试情绪系统
    es = EmotionSystem()

    test_inputs = [
        "今天天气真好，心情特别愉快！",
        "太让我意外了，居然通过了考试！",
        "最近遇到了一些烦心事，真的很郁闷。",
        "我恨死那个人了，简直不可理喻！",
        "嗯，挺好的，没什么特别的事。",
    ]

    print("=== 情绪状态机测试 ===\n")
    for inp in test_inputs:
        new_emotion = es.update_emotion(inp)
        print(f"输入: {inp}")
        print(f"新情绪: {new_emotion}")
        print(f"风格: {es.get_emotion_style_prompt()}")
        print(f"Prompt注入: {es.get_emotion_for_prompt()}")
        print()

    print(f"情绪历史: {len(es.emotion_history)} 次转换")
    print("测试完成。")
