import math
import re
from collections import Counter, defaultdict
from typing import Protocol

from app.recsys.schemas import ProductCandidate, RecUserProfile

# 领域关键词词表：帮助中文无分词器精准切出高区分度保险词根
_DOMAIN_TOKENS = (
    "保证续保20年",
    "保证续保",
    "外购药",
    "特需医疗",
    "高端医疗",
    "0免赔",
    "免赔额",
    "门急诊",
    "百万医疗",
    "防癌医疗",
    "防癌重疾",
    "少儿重疾",
    "少儿特疾",
    "少儿医疗",
    "少儿意外",
    "定期重疾",
    "终身重疾",
    "多次赔付",
    "定期寿险",
    "终身寿险",
    "增额终身寿",
    "高杠杆",
    "综合意外",
    "意外医疗",
    "老人意外",
    "猝死",
    "高危职业",
    "核保宽松",
    "免健告",
    "三高",
    "等待期",
    "家庭保障",
    "家庭支柱",
    "高性价比",
    "学平险",
    "中老年",
    "高龄",
    "女性",
    "投保人豁免",
)

_CATEGORY_TERMS = {
    "medical": "医疗险 住院 报销 医疗",
    "critical_illness": "重疾险 重大疾病 重疾 轻症 中症",
    "accident": "意外险 意外 伤残 猝死 骨折",
    "life": "寿险 身故 定期寿险 终身寿险 传承",
}


def tokenize_insurance_text(text: str) -> list[str]:
    """混合分词：领域专业词 + 英文数字词 + 中文二元字组（零外部依赖，离线可测）。"""
    lowered = text.lower()
    tokens: list[str] = []

    for term in _DOMAIN_TOKENS:
        if term.lower() in lowered:
            tokens.append(term.lower())

    for word in re.findall(r"[a-z0-9]+", lowered):
        tokens.append(word)

    for segment in re.findall(r"[\u4e00-\u9fa5]+", lowered):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            for i in range(len(segment) - 1):
                tokens.append(segment[i : i + 2])
    return tokens


def build_profile_query_text(profile: RecUserProfile) -> str:
    """将用户画像与原始自然语言查询组合为检索 Query。"""
    parts: list[str] = []
    if profile.query_text:
        parts.append(profile.query_text)
    if profile.age <= 17:
        parts.append("少儿 儿童 未成年")
    elif profile.age >= 50:
        parts.append("中老年 父母 高龄")
    if profile.occupation_class >= 5:
        parts.append("高危职业 1-6类职业")
    if not profile.has_social_security:
        parts.append("无社保 医保")
    for cat in profile.target_categories:
        if cat in _CATEGORY_TERMS:
            parts.append(_CATEGORY_TERMS[cat])
    if profile.family_role:
        parts.append(profile.family_role)
    parts.extend(profile.preferences)
    return " ".join(parts).strip() or "综合保险配置推荐"


class RecallChannel(Protocol):
    name: str

    def recall(
        self,
        profile: RecUserProfile,
        candidates: list[ProductCandidate],
        limit: int = 20,
    ) -> list[int]:
        ...


class BM25KeywordRecall:
    """内存 BM25 词法召回器（精准匹配险种术语、标签、亮点）"""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def recall(
        self,
        profile: RecUserProfile,
        candidates: list[ProductCandidate],
        limit: int = 20,
    ) -> list[int]:
        if not candidates:
            return []

        query_tokens = tokenize_insurance_text(build_profile_query_text(profile))
        if not query_tokens:
            return [c.id for c in candidates[:limit]]

        docs_tokens = [
            tokenize_insurance_text(
                f"{c.to_semantic_text()} {' '.join(c.coverage_tags * 2)} {' '.join(c.highlights)}"
            )
            for c in candidates
        ]
        n_docs = len(docs_tokens)
        avgdl = sum(len(d) for d in docs_tokens) / max(n_docs, 1)

        df: Counter[str] = Counter()
        tfs: list[Counter[str]] = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            tfs.append(tf)
            for token in tf:
                df[token] += 1

        scored: list[tuple[float, int]] = []
        for idx, cand in enumerate(candidates):
            doc_len = len(docs_tokens[idx])
            tf_map = tfs[idx]
            score = 0.0
            for q in query_tokens:
                if q not in tf_map:
                    continue
                idf = math.log((n_docs - df[q] + 0.5) / (df[q] + 0.5) + 1.0)
                freq = tf_map[q]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(avgdl, 1.0)))
                score += idf * (freq * (self.k1 + 1.0)) / max(denom, 1e-9)
            scored.append((score, -cand.id))

        scored.sort(reverse=True)
        return [-neg_id for _, neg_id in scored[:limit]]


class OfflineSemanticRecall:
    """离线 TF-IDF 余弦语义召回器（用于离线 TDD 与无 Milvus 环境的基线对比）"""

    name = "semantic"

    def recall(
        self,
        profile: RecUserProfile,
        candidates: list[ProductCandidate],
        limit: int = 20,
    ) -> list[int]:
        if not candidates:
            return []

        query_tokens = tokenize_insurance_text(build_profile_query_text(profile))
        docs_tokens = [tokenize_insurance_text(c.to_semantic_text()) for c in candidates]
        n_docs = len(docs_tokens)

        df: Counter[str] = Counter()
        for tokens in docs_tokens:
            for token in set(tokens):
                df[token] += 1

        def tfidf_vec(tokens: list[str]) -> dict[str, float]:
            tf = Counter(tokens)
            total = max(len(tokens), 1)
            return {
                tok: (cnt / total) * (math.log((n_docs + 1.0) / (df.get(tok, 0) + 1.0)) + 1.0)
                for tok, cnt in tf.items()
            }

        q_vec = tfidf_vec(query_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[tuple[float, int]] = []
        for idx, cand in enumerate(candidates):
            d_vec = tfidf_vec(docs_tokens[idx])
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_val * d_vec.get(tok, 0.0) for tok, q_val in q_vec.items())
            cos = dot / (q_norm * d_norm)
            scored.append((cos, -cand.id))

        scored.sort(reverse=True)
        return [-neg_id for _, neg_id in scored[:limit]]


class MilvusProductRecall:
    """线上 Milvus 产品目录语义召回器（查询 insurance_product_catalog，失败自动降级为离线语义召回）"""

    name = "milvus_semantic"

    def __init__(
        self,
        vector_store: object | None = None,
        collection_name: str = "insurance_product_catalog",
    ) -> None:
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._fallback = OfflineSemanticRecall()

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        from langchain_milvus import BM25BuiltInFunction, Milvus
        from langchain_openai import OpenAIEmbeddings

        from app.core.config import settings

        embeddings = OpenAIEmbeddings(
            model="text-embedding-v4",
            api_key=settings.llm.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            dimensions=1024,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )
        self._vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=self._collection_name,
            builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
            vector_field=["dense", "sparse"],
            connection_args={"uri": settings.rag.milvus_url},
            auto_id=True,
        )
        return self._vector_store

    def recall(
        self,
        profile: RecUserProfile,
        candidates: list[ProductCandidate],
        limit: int = 20,
    ) -> list[int]:
        if not candidates:
            return []

        valid_ids = {c.id for c in candidates}
        query_text = build_profile_query_text(profile)
        try:
            vs = self._get_vector_store()
            docs = vs.similarity_search(query=query_text, k=max(limit * 2, 20))
            seen: set[int] = set()
            ordered_ids: list[int] = []
            for doc in docs:
                pid_raw = (getattr(doc, "metadata", None) or {}).get("product_id")
                if pid_raw is None:
                    continue
                pid = int(pid_raw)
                if pid in valid_ids and pid not in seen:
                    seen.add(pid)
                    ordered_ids.append(pid)
                if len(ordered_ids) >= limit:
                    break
            if ordered_ids:
                return ordered_ids
        except Exception as exc:
            from app.core.logging import get_logger

            get_logger(__name__).warning("Milvus产品目录召回失败，降级离线语义召回", error=str(exc))

        return self._fallback.recall(profile, candidates, limit=limit)


class StructuredRuleRecall:
    """结构化规则召回器：按目标险种、软年龄窗口与预算带召回"""

    name = "structured"

    def recall(
        self,
        profile: RecUserProfile,
        candidates: list[ProductCandidate],
        limit: int = 20,
    ) -> list[int]:
        if not candidates:
            return []

        target_cats = set(profile.target_categories)
        scored: list[tuple[float, int]] = []
        for cand in candidates:
            score = 0.0
            if not target_cats or cand.category in target_cats:
                score += 3.0
            # 软年龄窗口，防止边界年龄被误丢，硬剔除留给 Stage 2
            if (cand.min_age - 2) <= profile.age <= (cand.max_age + 2):
                score += 2.0
            if profile.occupation_class <= cand.max_occupation_class:
                score += 1.5
            if profile.annual_budget > 0 and cand.min_premium <= profile.annual_budget:
                score += 1.0
            if cand.is_featured:
                score += 0.3
            scored.append((score, -cand.id))

        scored.sort(reverse=True)
        return [-neg_id for _, neg_id in scored[:limit]]


def rrf_fuse(
    ranked_lists: dict[str, list[int]],
    k: int = 60,
    top_k: int = 15,
    category_map: dict[int, str] | None = None,
    target_categories: list[str] | None = None,
    min_per_category: int = 3,
) -> list[tuple[int, float, dict[str, int]]]:
    """倒数排名融合（Reciprocal Rank Fusion），支持多险种配额防单品类垄断。"""
    rrf_scores: dict[int, float] = defaultdict(float)
    recall_ranks: dict[int, dict[str, int]] = defaultdict(dict)

    for channel_name, ids in ranked_lists.items():
        for rank, pid in enumerate(ids, start=1):
            rrf_scores[pid] += 1.0 / (k + rank)
            recall_ranks[pid][channel_name] = rank

    ordered = sorted(
        rrf_scores.keys(),
        key=lambda pid: (-rrf_scores[pid], pid),
    )

    # 若指定了多险种，先按险种保底取出各自高分候选，再按全局 RRF 分补齐至 top_k
    if category_map and target_categories and len(target_categories) > 1:
        selected_ids: list[int] = []
        seen: set[int] = set()
        for cat in dict.fromkeys(target_categories):
            taken = 0
            for pid in ordered:
                if pid in seen:
                    continue
                if category_map.get(pid) == cat:
                    selected_ids.append(pid)
                    seen.add(pid)
                    taken += 1
                    if taken >= min_per_category:
                        break
        for pid in ordered:
            if len(selected_ids) >= top_k:
                break
            if pid not in seen:
                selected_ids.append(pid)
                seen.add(pid)
        final_ids = selected_ids[:top_k]
    else:
        final_ids = ordered[:top_k]

    return [(pid, round(rrf_scores[pid], 6), dict(recall_ranks[pid])) for pid in final_ids]
