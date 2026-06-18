"""
Step 3.x — ChromaDB 长期记忆系统
包括记忆写入、RAG 检索、记忆管理策略
"""
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

load_dotenv()


class MemorySystem:
    """基于 ChromaDB 的长期记忆系统"""

    def __init__(self, persist_dir: str = None, collection_name: str = "companion_memories"):
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./chroma_db")

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
 
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── 写入 ─────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        memory_type: str = "summary",
        emotion: str = "平静",
        importance: int = 5,
        metadata: Optional[Dict] = None,
    ) -> str:
        """添加一条记忆到 ChromaDB"""
        mem_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        meta = {
            "type": memory_type,
            "emotion": emotion,
            "importance": importance,
            "timestamp": timestamp,
        }
        if metadata:
            meta.update(metadata)

        self.collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[meta],
        )
        return mem_id

    def add_conversation_summary(self, summary: str, emotion: str = "平静", importance: int = 5) -> str:
        """添加一段对话摘要作为记忆"""
        return self.add_memory(
            content=summary,
            memory_type="summary",
            emotion=emotion,
            importance=importance,
        )

    def add_key_event(self, event: str, emotion: str, importance: int = 7) -> str:
        """记录一个关键事件（用户提到的重要事情）"""
        return self.add_memory(
            content=event,
            memory_type="event",
            emotion=emotion,
            importance=importance,
        )

    def add_user_fact(self, fact: str, importance: int = 6) -> str:
        """记录一个关于用户的事实"""
        return self.add_memory(
            content=fact,
            memory_type="fact",
            importance=importance,
        )

    # ─── 检索 ─────────────────────────────────────────────

    def retrieve_relevant(
        self,
        query: str,
        n_results: int = 5,
        threshold: float = 0.3,
        importance_min: int = 1,
    ) -> List[Dict]:
        """检索与查询相关的记忆"""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance = 1 - distance  # cosine distance -> similarity

                # 过滤：低于相似度阈值 或 重要性不够
                if relevance < threshold:
                    continue
                if meta.get("importance", 5) < importance_min:
                    continue

                memories.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "relevance": round(relevance, 3),
                    "importance": meta.get("importance", 5),
                    "type": meta.get("type", "unknown"),
                    "emotion": meta.get("emotion", "平静"),
                    "timestamp": meta.get("timestamp", ""),
                })

        # 按重要性降序排列
        memories.sort(key=lambda m: (m["importance"], m["relevance"]), reverse=True)
        return memories

    def format_memories_for_prompt(self, memories: List[Dict], max_chars: int = 500) -> str:
        """将检索到的记忆格式化成 Prompt 可用的文本"""
        if not memories:
            return "暂无相关记忆。"

        lines = []
        chars = 0
        for m in memories:
            entry = f"- [{m['type']}] {m['content']} (相关度: {m['relevance']:.2f})"
            if chars + len(entry) > max_chars:
                break
            lines.append(entry)
            chars += len(entry)
        return "\n".join(lines)

    # ─── 管理 ─────────────────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        """获取所有记忆"""
        if self.collection.count() == 0:
            return []

        results = self.collection.get()
        memories = []
        for i, doc in enumerate(results["documents"]):
            memories.append({
                "id": results["ids"][i],
                "content": doc,
                **results["metadatas"][i],
            })
        return memories

    def delete_low_importance_memories(self, threshold: int = 3):
        """遗忘机制：清理低优先级记忆"""
        all_memories = self.get_all_memories()
        to_delete = [
            m["id"] for m in all_memories
            if m.get("importance", 5) < threshold
        ]
        if to_delete:
            self.collection.delete(ids=to_delete)
            return len(to_delete)
        return 0

    def count(self) -> int:
        """当前记忆总数"""
        return self.collection.count()

    def clear_all(self):
        """清空所有记忆"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)


if __name__ == "__main__":
    # 简单测试
    import shutil
    test_dir = "./chroma_db_test"

    # 清理旧的测试数据
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    ms = MemorySystem(persist_dir=test_dir)

    # 写入测试记忆
    ms.add_conversation_summary("用户说最近在学习 Python，对数据科学很感兴趣。", importance=6)
    ms.add_key_event("用户提到下周有一个重要的面试。", emotion="紧张", importance=9)
    ms.add_user_fact("用户喜欢喝咖啡，每天早上都会泡一杯。", importance=4)
    ms.add_conversation_summary("用户今天心情很好，刚收到一个好消息。", emotion="愉悦")

    print(f"记忆总数: {ms.count()}")

    # 检索测试
    results = ms.retrieve_relevant("用户的学习情况", n_results=5)
    print(f"\n检索 '用户的学习情况':")
    for r in results:
        print(f"  [{r['type']}] {r['content']} (重要性:{r['importance']}, 相关度:{r['relevance']})")

    results2 = ms.retrieve_relevant("面试和工作", n_results=5)
    print(f"\n检索 '面试和工作':")
    for r in results2:
        print(f"  [{r['type']}] {r['content']} (重要性:{r['importance']}, 相关度:{r['relevance']})")

    # 格式化成 Prompt
    print(f"\n--- Prompt 格式化输出 ---")
    print(ms.format_memories_for_prompt(results2))

    # 清理
    shutil.rmtree(test_dir)
    print(f"\n测试完成。")
