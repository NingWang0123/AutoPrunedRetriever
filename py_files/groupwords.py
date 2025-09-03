# semantic_merge_graph_demo.py
# 在“形态归一合并”的基础上，再做“语义同义合并”（embedding + kNN + Union-Find）
# 依赖：pip install sentence-transformers faiss-cpu networkx nltk numpy

import re
import unicodedata
from collections import defaultdict
import networkx as nx
import numpy as np
import faiss
from langchain.embeddings.base import Embeddings
from gensim.models import KeyedVectors

class WordAvgEmbeddings(Embeddings):
    def __init__(self, model_path: str = "gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model.vectors.npy"):
        self.kv = KeyedVectors.load(model_path, mmap='r')
        self.dim = self.kv.vector_size
        self.token_pat = re.compile(r"[A-Za-z]+")

    def _embed_text(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in self.token_pat.findall(text)]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        return self._embed_text(text)

# ------- (可选) NLTK 的 wordnet 资源仅用于 lemmatizer 时；本 demo 未用 lemmatizer，无需下载 -------
word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")
# ============== 形态规范化（与之前一致，关闭词干，保留可读词形） ==============
_LANG = "english"
_STEM = False  # 不做词干化，避免 'require' -> 'requir'
_STOPWORDS = {
    "the","a","an","is","are","was","were","am","be","been","being",
    "of","to","in","import","on","at","for","with","by","and","or","as","that",
    "this","these","those","it","its","from","into","than","then","so",
}
_TOKEN_PAT = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]+|\d+")

def normalize_text(text: str) -> str:
    """NFKC→lower→分词→去停用→(可选词干)→空格连接。保持可读词形。"""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    toks = _TOKEN_PAT.findall(t)
    out = []
    for tok in toks:
        # 英文：去停用；中文/数字：保留
        if tok.isalpha() and tok.encode("utf-8").isalpha():
            if tok in _STOPWORDS:
                continue
            # 不做词干化，避免残根
            out.append(tok)
        else:
            out.append(tok)
    return " ".join(out).strip()

# ============== 工具：打印图 ==============
def print_graph(title: str, G: nx.Graph):
    print(f"\n=== {title} ===")
    print("Nodes:")
    for n, d in G.nodes(data=True):
        print(f"  - {n!r} :: {d}")
    print("Edges:")
    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            print(f"  - {u!r} --[{k}]--> {v!r} :: {d}")
    else:
        arrow = "->" if G.is_directed() else "--"
        for u, v, d in G.edges(data=True):
            print(f"  - {u!r} {arrow} {v!r} :: {d}")

# ============== 第一步：形态归一 + 合并 ==============
def merge_graph_nodes_by_canonical(
    G: nx.Graph,
    *,
    normalizer=normalize_text,
    node_alias_attr: str = "aliases",
    drop_self_loops: bool = True,
    edge_weight_key: str = "weight",
    merge_edge_attrs: tuple = ("relation", "label"),
):
    node2canon = {n: normalizer(n) for n in G.nodes()}
    canon_aliases = defaultdict(set)
    for n, c in node2canon.items():
        canon_aliases[c if c else str(n)].add(n)

    H = type(G)()
    H.graph.update(G.graph)

    for c, aliases in canon_aliases.items():
        H.add_node(c)
        H.nodes[c][node_alias_attr] = sorted(map(str, aliases))

    edge_bucket = {}

    def _edge_key(cu, cv):
        if G.is_directed():
            return (cu, cv)
        return tuple(sorted((cu, cv)))

    edge_iter = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for e in edge_iter:
        if G.is_multigraph():
            u, v, k, data = e
        else:
            u, v, data = e
        cu = node2canon[u] if node2canon[u] else str(u)
        cv = node2canon[v] if node2canon[v] else str(v)
        if cu == cv and drop_self_loops:
            continue

        key = _edge_key(cu, cv)
        if key not in edge_bucket:
            edge_bucket[key] = {edge_weight_key: 0.0 if edge_weight_key in data else None}
            for attr in merge_edge_attrs:
                edge_bucket[key][attr] = set()

        if edge_weight_key in data:
            if edge_bucket[key][edge_weight_key] is None:
                edge_bucket[key][edge_weight_key] = float(data[edge_weight_key])
            else:
                edge_bucket[key][edge_weight_key] += float(data[edge_weight_key])

        for attr in merge_edge_attrs:
            if attr in data and data[attr] is not None:
                val = data[attr]
                if isinstance(val, (list, set, tuple)):
                    edge_bucket[key][attr].update(map(str, val))
                else:
                    edge_bucket[key][attr].add(str(val))

    for (cu, cv), agg in edge_bucket.items():
        attrs = {}
        if agg.get(edge_weight_key, None) is not None:
            attrs[edge_weight_key] = agg[edge_weight_key]
        for attr in merge_edge_attrs:
            vals = sorted(agg.get(attr, set()))
            if vals:
                attrs[attr] = vals
        H.add_edge(cu, cv, **attrs)

    return H

# ============== 第二步：语义同义聚合（embedding + kNN + 并查集） ==============
class DSU:
    def __init__(self, n): self.p=list(range(n))
    def find(self, x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self, a,b):
        ra, rb = self.find(a), self.find(b)
        if ra!=rb: self.p[rb]=ra

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_embeddings(model_name: str, strings: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(strings, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    return l2_normalize(embs)  # 归一化后用内积≈余弦

def semantic_clusters_via_knn(strings: list, model_name: str, top_k=5, sim_threshold=0.82):
    """
    对字符串列表做 embedding，使用 FAISS 内积 kNN 找相似对（>=阈值）并并查集合并。
    返回：clusters: List[List[int]]，每个簇是原索引的集合
    """
    if len(strings) == 0:
        return []

    X = build_embeddings(model_name, strings)   # (n, d) 已归一化
    n, d = X.shape
    index = faiss.IndexFlatIP(d)
    index.add(X)
    # 自身最近邻是自己，取 top_k+1 再排除 self
    D, I = index.search(X, min(top_k+1, n))

    dsu = DSU(n)
    for i in range(n):
        for j_idx, sim in zip(I[i], D[i]):
            if j_idx == i:
                continue
            if sim >= sim_threshold:
                dsu.union(i, j_idx)

    # 聚簇
    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values())

def pick_readable_name(candidates: list) -> str:
    """
    从一组别名中挑选“可读且稳定”的名称：
    - 优先长度较短、字母数字为主，其次字典序
    """
    def score(s: str):
        # 较短更好；尽量少非字母数字字符
        non_alnum = sum(1 for ch in s if not ch.isalnum() and ch != ' ')
        return (len(s), non_alnum, s.lower())
    return sorted(candidates, key=score)[0]

def merge_graph_nodes_by_semantic(
    G: nx.Graph,
    *,
    normalizer=normalize_text,
    node_alias_attr: str = "aliases",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    sim_threshold: float = 0.82,
    top_k: int = 5,
    drop_self_loops: bool = True,
    edge_weight_key: str = "weight",
    merge_edge_attrs: tuple = ("relation","label"),
):
    """
    先做形态归一合并，再对规范化后的节点名做语义聚类，最后以“代表名”落盘并合并边。
    """
    # ① 形态归一合并
    H = merge_graph_nodes_by_canonical(
        G,
        normalizer=normalizer,
        node_alias_attr=node_alias_attr,
        drop_self_loops=drop_self_loops,
        edge_weight_key=edge_weight_key,
        merge_edge_attrs=merge_edge_attrs,
    )

    # ② 取出规范化后的节点名（H 的节点键就是 canonical）
    canon_nodes = list(H.nodes())
    if len(canon_nodes) <= 1:
        return H

    # ③ 对 canonical 名称做向量，并通过 kNN+并查集得到语义簇
    clusters = semantic_clusters_via_knn(
        strings=canon_nodes,
        model_name=model_name,
        top_k=top_k,
        sim_threshold=sim_threshold,
    )

    # ④ 为每个簇挑选一个“显示名”（优先从别名里挑最可读）
    cluster_repr = {}
    for cluster in clusters:
        # 收集该簇内全部别名（原始文本）
        all_aliases = []
        for idx in cluster:
            cnode = canon_nodes[idx]
            all_aliases.extend(H.nodes[cnode].get(node_alias_attr, [cnode]))
        rep = pick_readable_name(all_aliases)  # 代表名
        cluster_repr[tuple(sorted(cluster))] = rep

    # ⑤ 构造映射：canonical -> 代表名
    canon_to_rep = {}
    for cluster in clusters:
        rep = cluster_repr[tuple(sorted(cluster))]
        for idx in cluster:
            canon_to_rep[canon_nodes[idx]] = rep

    # ⑥ 重新构图：把节点重命名为代表名，并合并边
    Z = type(H)()
    Z.graph.update(H.graph)

    # 先把代表名节点建好，aliases 汇总（包含所有原 alias）
    rep_aliases = defaultdict(set)
    for cnode in H.nodes():
        rep = canon_to_rep.get(cnode, cnode)
        rep_aliases[rep].update(H.nodes[cnode].get(node_alias_attr, [cnode]))
    for rep, aliases in rep_aliases.items():
        Z.add_node(rep)
        Z.nodes[rep][node_alias_attr] = sorted(set(aliases))

    # 聚合边
    edge_bucket = {}
    def _edge_key(u,v):
        if Z.is_directed():
            return (u,v)
        return tuple(sorted((u,v)))

    edge_iter = H.edges(keys=True, data=True) if H.is_multigraph() else H.edges(data=True)
    for e in edge_iter:
        if H.is_multigraph():
            u, v, k, data = e
        else:
            u, v, data = e
        ru, rv = canon_to_rep.get(u, u), canon_to_rep.get(v, v)
        if ru == rv and drop_self_loops:
            continue
        key = _edge_key(ru, rv)
        if key not in edge_bucket:
            edge_bucket[key] = {edge_weight_key: 0.0 if edge_weight_key in data else None}
            for attr in merge_edge_attrs:
                edge_bucket[key][attr] = set()

        if edge_weight_key in data:
            if edge_bucket[key][edge_weight_key] is None:
                edge_bucket[key][edge_weight_key] = float(data[edge_weight_key])
            else:
                edge_bucket[key][edge_weight_key] += float(data[edge_weight_key])

        for attr in merge_edge_attrs:
            if attr in data and data[attr] is not None:
                val = data[attr]
                if isinstance(val, (list, set, tuple)):
                    edge_bucket[key][attr].update(map(str, val))
                else:
                    edge_bucket[key][attr].add(str(val))

    for (u, v), agg in edge_bucket.items():
        attrs = {}
        if agg.get(edge_weight_key, None) is not None:
            attrs[edge_weight_key] = agg[edge_weight_key]
        for attr in merge_edge_attrs:
            vals = sorted(agg.get(attr, set()))
            if vals:
                attrs[attr] = vals
        Z.add_edge(u, v, **attrs)

    return Z
# ============== （新增）第三步：用 Word Embedding 做语义同义聚合（对比实验） ==============
# 思路：对节点名分词 -> 取每个 token 的词向量 -> 求平均作为节点向量
#       然后与上面的 Sentence 版本一样，用 FAISS + kNN + Union-Find 合并
# 说明：大多数预训练词向量是英文语料，对中文/混合文本支持较弱；这是本对比实验想要展示的点。

def _load_gensim_kv(preferred_models=("glove-wiki-gigaword-100", "glove-wiki-gigaword-50"), local_path: str = None):
    """
    加载 gensim 的 KeyedVectors：
    - 若提供 local_path（.kv/.bin/.txt 等），优先走本地加载；
    - 否则尝试通过 gensim.downloader 下载指定的英文模型（需要联网）。
    返回：KeyedVectors
    """
    from gensim.models import KeyedVectors
    try:
        if local_path:
            # 自动识别文本/二进制
            return KeyedVectors.load(local_path, mmap='r')
    except Exception:
        # 如果是 word2vec/text 格式也可用 KeyedVectors.load_word2vec_format
        try:
            return KeyedVectors.load_word2vec_format(local_path, binary=local_path.endswith(".bin"))
        except Exception as e:
            print(f"[word-emb] Failed to load local model from {local_path}: {e}")

    # 尝试在线下载（如环境允许）
    try:
        import gensim.downloader as api
        for name in preferred_models:
            try:
                print(f"[word-emb] trying to load '{name}' via gensim.downloader ...")
                kv = api.load(name)  # 例如 'glove-wiki-gigaword-100'
                print(f"[word-emb] loaded '{name}'")
                return kv
            except Exception as e:
                print(f"[word-emb] failed '{name}': {e}")
    except Exception as e:
        print(f"[word-emb] gensim.downloader not available: {e}")

    raise RuntimeError("No word embedding model could be loaded. "
                       "Provide a local_path or enable internet for gensim.downloader.")

def _simple_english_tokenize(s: str):
    """
    很简单的英文 token 切分（与 normalize_text 的分词风格保持一致）。
    只取纯字母 token，用于词向量查询；中文会被忽略（多数英文词向量无中文）。
    """
    toks = _TOKEN_PAT.findall(s.lower())
    return [t for t in toks if t.isalpha()]

def build_word_embeddings(strings: list, kv=None, local_path: str = None) -> np.ndarray:
    """
    用词向量为每个字符串生成一个向量（平均词向量）。
    - OOV token 会被跳过；若全部 OOV，则回退为零向量（稍后会被丢到单位化里成为0，需处理）。
    """
    if kv is None:
        kv = _load_gensim_kv(local_path=local_path)
    dim = kv.vector_size
    X = np.zeros((len(strings), dim), dtype=np.float32)

    for i, s in enumerate(strings):
        toks = _simple_english_tokenize(s)
        vecs = []
        for tok in toks:
            if tok in kv:
                vecs.append(kv[tok])
        if len(vecs) > 0:
            X[i] = np.mean(vecs, axis=0)
        else:
            X[i] = 0.0  # 全部 OOV 的情况

    # 单位化；全零向量的行会被保持为零（避免 nan）
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    X = X / n
    return X

def semantic_clusters_via_knn_word(strings: list, kv=None, local_path: str = None, top_k=5, sim_threshold=0.82):
    """
    使用 Word Embedding（平均词向量）做 kNN + Union-Find 聚类。
    返回：clusters: List[List[int]]
    """
    if len(strings) == 0:
        return []
    X = build_word_embeddings(strings, kv=kv, local_path=local_path)
    n, d = X.shape
    index = faiss.IndexFlatIP(d)
    index.add(X)
    D, I = index.search(X, min(top_k+1, n))

    dsu = DSU(n)
    for i in range(n):
        for j_idx, sim in zip(I[i], D[i]):
            if j_idx == i:
                continue
            if sim >= sim_threshold:
                dsu.union(i, j_idx)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values())

def merge_graph_nodes_by_semantic_word(
    G: nx.Graph,
    *,
    normalizer=normalize_text,
    node_alias_attr: str = "aliases",
    kv=None,
    local_path: str = None,  # 你也可以传入本地词向量路径
    sim_threshold: float = 0.82,
    top_k: int = 5,
    drop_self_loops: bool = True,
    edge_weight_key: str = "weight",
    merge_edge_attrs: tuple = ("relation","label"),
):
    """
    用 Word Embedding（平均词向量）进行语义聚合。
    （流程与 merge_graph_nodes_by_semantic 基本相同，只是 embedding 构建方式不同）
    """
    # ① 形态归一合并
    H = merge_graph_nodes_by_canonical(
        G,
        normalizer=normalizer,
        node_alias_attr=node_alias_attr,
        drop_self_loops=drop_self_loops,
        edge_weight_key=edge_weight_key,
        merge_edge_attrs=merge_edge_attrs,
    )

    canon_nodes = list(H.nodes())
    if len(canon_nodes) <= 1:
        return H

    # ② 词向量聚类（平均词向量）
    clusters = semantic_clusters_via_knn_word(
        strings=canon_nodes,
        kv=kv,
        local_path=local_path,
        top_k=top_k,
        sim_threshold=sim_threshold,
    )

    # ③ 代表名选择与映射（沿用上面 Sentence 版的逻辑）
    cluster_repr = {}
    for cluster in clusters:
        all_aliases = []
        for idx in cluster:
            cnode = canon_nodes[idx]
            all_aliases.extend(H.nodes[cnode].get(node_alias_attr, [cnode]))
        rep = pick_readable_name(all_aliases)
        cluster_repr[tuple(sorted(cluster))] = rep

    canon_to_rep = {}
    for cluster in clusters:
        rep = cluster_repr[tuple(sorted(cluster))]
        for idx in cluster:
            canon_to_rep[canon_nodes[idx]] = rep

    Z = type(H)()
    Z.graph.update(H.graph)

    rep_aliases = defaultdict(set)
    for cnode in H.nodes():
        rep = canon_to_rep.get(cnode, cnode)
        rep_aliases[rep].update(H.nodes[cnode].get(node_alias_attr, [cnode]))
    for rep, aliases in rep_aliases.items():
        Z.add_node(rep)
        Z.nodes[rep][node_alias_attr] = sorted(set(aliases))

    edge_bucket = {}
    def _edge_key(u,v):
        if Z.is_directed():
            return (u,v)
        return tuple(sorted((u,v)))

    edge_iter = H.edges(keys=True, data=True) if H.is_multigraph() else H.edges(data=True)
    for e in edge_iter:
        if H.is_multigraph():
            u, v, k, data = e
        else:
            u, v, data = e
        ru, rv = canon_to_rep.get(u, u), canon_to_rep.get(v, v)
        if ru == rv and drop_self_loops:
            continue
        key = _edge_key(ru, rv)
        if key not in edge_bucket:
            edge_bucket[key] = {edge_weight_key: 0.0 if edge_weight_key in data else None}
            for attr in merge_edge_attrs:
                edge_bucket[key][attr] = set()

        if edge_weight_key in data:
            if edge_bucket[key][edge_weight_key] is None:
                edge_bucket[key][edge_weight_key] = float(data[edge_weight_key])
            else:
                edge_bucket[key][edge_weight_key] += float(data[edge_weight_key])

        for attr in merge_edge_attrs:
            if attr in data and data[attr] is not None:
                val = data[attr]
                if isinstance(val, (list, set, tuple)):
                    edge_bucket[key][attr].update(map(str, val))
                else:
                    edge_bucket[key][attr].add(str(val))

    for (u, v), agg in edge_bucket.items():
        attrs = {}
        if agg.get(edge_weight_key, None) is not None:
            attrs[edge_weight_key] = agg[edge_weight_key]
        for attr in merge_edge_attrs:
            vals = sorted(agg.get(attr, set()))
            if vals:
                attrs[attr] = vals
        Z.add_edge(u, v, **attrs)

    return Z

# ============== Demo 构图：加入语义同义情况（英文+中文别名） ==============
def build_semantic_sample_undirected() -> nx.Graph:
    G = nx.Graph()
    # 这些将被形态合并为 ("cat house")，语义层不再变化
    G.add_node("A Cat in the House")
    G.add_node("the cats in a house")

    # 这些是语义同义（组织/银行），希望被合并：
    G.add_node("ICBC")
    G.add_node("Industrial and Commercial Bank of China")


    # 另一个语义同义组：
    G.add_node("Google DeepMind")
    G.add_node("DeepMind")
    G.add_node("Google  Deep  Mind")

    # 对边加点联系
    G.add_edge("A Cat in the House", "ICBC", relation="mentioned_with", weight=1.0)
    G.add_edge("Google DeepMind", "Industrial and Commercial Bank of China", relation="research_vs_finance", weight=1.5)
    G.add_edge("DeepMind", "ICBC", relation="misc", weight=0.8)
    return G

def build_semantic_sample_directed() -> nx.DiGraph:
    D = nx.DiGraph()
    D.add_node("OpenAI")
    D.add_node("Open AI")  # 形态上会归一到 open ai
    D.add_node("Anthropic")
    D.add_node("Claude company")  # 粗略语义相关（不一定合并，取决于阈值）

    D.add_edge("OpenAI", "Anthropic", relation="compete_with", weight=1.0)
    D.add_edge("Open AI", "Anthropic", relation="compete_with", weight=2.0)
    D.add_edge("Claude company", "OpenAI", relation="related_to", weight=1.0)
    return D

def main():
    # ===== 无向图：银行 / DeepMind 同义测试 =====
    G0 = build_semantic_sample_undirected()
    print_graph("Undirected Graph (Before)", G0)

    # 仅形态合并
    G_canon = merge_graph_nodes_by_canonical(G0, normalizer=normalize_text, merge_edge_attrs=("relation",))
    print_graph("Undirected Graph (After Canonical Merge)", G_canon)

    # Sentence Embedding 语义合并
    G_sem_sent = merge_graph_nodes_by_semantic(
        G0,
        normalizer=normalize_text,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        sim_threshold=0.80,
        top_k=5,
        merge_edge_attrs=("relation",),
    )
    print_graph("Undirected Graph (Sentence Embedding Semantic Merge)", G_sem_sent)

    # Word Embedding 语义合并（对比）
    # 如需指定本地词向量：local_path="/path/to/your/kv-or-bin"
    G_sem_word = merge_graph_nodes_by_semantic_word(
        G0,
        normalizer=normalize_text,
        kv=word_emb.kv,          # 若有本地模型可填路径；否则会尝试 gensim.downloader
        sim_threshold=0.80,
        top_k=5,
        merge_edge_attrs=("relation",),
    )
    print_graph("Undirected Graph (Word Embedding Semantic Merge)", G_sem_word)

    # ===== 有向图：OpenAI / Open AI =====
    D0 = build_semantic_sample_directed()
    print_graph("Directed Graph (Before)", D0)

    D_canon = merge_graph_nodes_by_canonical(D0, normalizer=normalize_text, merge_edge_attrs=("relation",))
    print_graph("Directed Graph (After Canonical Merge)", D_canon)

    D_sem_sent = merge_graph_nodes_by_semantic(
        D0,
        normalizer=normalize_text,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        sim_threshold=0.82,
        top_k=4,
        merge_edge_attrs=("relation",),
    )
    print_graph("Directed Graph (Sentence Embedding Semantic Merge)", D_sem_sent)

    D_sem_word = merge_graph_nodes_by_semantic_word(
        D0,
        normalizer=normalize_text,
        kv=word_emb.kv, 
        sim_threshold=0.82,
        top_k=4,
        merge_edge_attrs=("relation",),
    )
    print_graph("Directed Graph (Word Embedding Semantic Merge)", D_sem_word)

if __name__ == "__main__":
    main()
