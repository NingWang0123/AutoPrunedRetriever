import re
from typing import List, Optional,Iterable,Any,Callable, Set, Union
import numpy as np
from gensim.models import KeyedVectors
import numpy as np
import re
from langchain.embeddings.base import Embeddings
import gensim.downloader as api

class Word2VecEmbeddings(Embeddings):
    """
    与你现有 WordAvgEmbeddings 相同接口的静态词向量平均器：
    - 可用 gensim.downloader 加载（model_name）
    - 或本地路径加载（.model / .bin / .txt / word2vec 格式）
    - 或直接传入已加载的 KeyedVectors (kv)
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        kv: Optional[KeyedVectors] = None,
        l2_normalize: bool = True,
        token_pattern: str = r"[A-Za-z]+"
    ):
        if kv is not None:
            self.kv = kv
        elif model_name is not None:
            # 例如: "word2vec-google-news-300" / "glove-wiki-gigaword-100"
            self.kv = api.load(model_name)
        elif model_path:
            # 先尝试 KeyedVectors 原生保存格式 .model
            try:
                self.kv = KeyedVectors.load(model_path, mmap='r')
            except Exception:
                # 回退到 word2vec 文本/二进制格式
                binary = str(model_path).lower().endswith((".bin", ".gz"))
                self.kv = KeyedVectors.load_word2vec_format(model_path, binary=binary)
        else:
            raise ValueError("Provide one of: `kv`, `model_name`, or `model_path`.")

        self.dim = int(self.kv.vector_size)
        self.l2_normalize = bool(l2_normalize)
        self.token_pat = re.compile(token_pattern)

    # ---- 内部：单条文本向量化（与原类同名同义）----
    def _embed_text(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in self.token_pat.findall(text or "")]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            v = np.zeros(self.dim, dtype=np.float32)
        else:
            v = np.mean(vecs, axis=0).astype(np.float32)
        if self.l2_normalize:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v

    # ---- LangChain 需要的接口：多文档 ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t).tolist() for t in texts]

    # ---- LangChain 需要的接口：单查询 ----
    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text).tolist()

class WordAvgEmbeddings(Embeddings):
    """
    A simple word-averaging embedding for LangChain.
    - 输入文本会被正则分词（只保留 a-zA-Z），并转小写
    - 每个词用 KeyedVectors 查词向量，取均值
    - OOV 时返回全零向量
    - 支持可选 L2 归一化，返回 Python list（FAISS 友好）
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        kv: Optional[KeyedVectors] = None,
        l2_normalize: bool = True,
        token_pattern: str = r"[A-Za-z]+"
    ):
        """
        Args:
            model_path: 本地 KeyedVectors 路径（.kv / .bin / word2vec 格式）。
                        例如：'gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model'
            kv:         也可以直接传入已加载的 KeyedVectors（与 model_path 二选一）
            l2_normalize: 是否对平均向量做 L2 归一化
            token_pattern: 分词正则
        """
        if kv is not None:
            self.kv = kv
        elif model_path:
            # 尝试用 KeyedVectors.load 加载；失败则回退到 word2vec 格式加载
            try:
                self.kv = KeyedVectors.load(model_path, mmap='r')
            except Exception:
                # 如果是 word2vec / text 格式
                self.kv = KeyedVectors.load_word2vec_format(model_path, binary=False)
        else:
            raise ValueError("Provide either `model_path` or `kv`.")

        self.dim = self.kv.vector_size
        self.l2_normalize = l2_normalize
        self.token_pat = re.compile(token_pattern)

    # ---- 内部：单条文本向量化 ----
    def _embed_text(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in self.token_pat.findall(text)]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            v = np.zeros(self.dim, dtype=np.float32)
        else:
            v = np.mean(vecs, axis=0).astype(np.float32)
        if self.l2_normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    # ---- LangChain 抽象方法：多文档 ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain 期望返回 List[List[float]]
        out = [self._embed_text(t).tolist() for t in texts]
        return out

    # ---- LangChain 抽象方法：单查询 ----
    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text).tolist()
    