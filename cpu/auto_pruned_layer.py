"""
AutoPruned Layer (APL) v3 — A plug-in module that enhances any RAG system.

Takes another RAG system's retrieved context, converts to structured KG format,
and generates answers using accumulated answer memory with proper retrieval,
find_related_knowledge, compact_indicies_for_prompt, slice_for_final_merged_json,
and combine_ents.

Memory: Answers and thinkings from previous Q&A are stored.
RAG context: Parsed into KG triples per-question, used in LLM prompt, then discarded.

Usage:
    python autopruned_layer.py \
        --predictions path/to/baseline_predictions.json \
        --output path/to/apl_output.json \
        --llm_api $OPENAI_API_KEY
"""

import json
import os
import sys
import time
import argparse
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import partial

# Import from cpu codebase
from AutoPrunedRetriever_advanced_cached import (
    get_code_book,
    merging_codebook,
    get_word_embeddings,
    decode_questions,
    select_best_context_by_keys,
    slice_for_final_merged_json,
    get_all_results_entire_chunk,
    get_all_results,
    get_json_with_given_knowledge,
    get_json_with_given_knowledge_and_thinkings,
    get_flat_answers_lsts,
    get_thinkings_lsts,
    is_no_answer_text,
)
from retrieve_gpu_boosted import coarse_filter_optimized_gpu_ver as coarse_filter_torch
from sentence_embed_overlap import get_unique_or_overlap_by_sentence_embedded
from graph_generator.llm_parser import TOKEN_STATS
from llm_api import OpenAILLM

# Import combine_ents
try:
    from combine_ent_cached import combine_ents_ann_knn
    _HAS_COMBINE = True
except ImportError:
    _HAS_COMBINE = False


class AutoPrunedLayer:
    """
    Plug-in layer that converts any RAG's retrieved context into
    structured KG format and maintains answer memory across questions.

    Matches the original run_work_flow pipeline:
    1. Parse RAG context → temporary facts KG
    2. Parse question → question KG
    3. Merge question into meta_codebook
    4. Retrieve answers (+ thinkings) from memory
    5. find_related_knowledge — filter/dedup via sentence embedding overlap
    6. compact_indicies_for_prompt — build final JSON with proper index remapping
    7. slice_for_final_merged_json → LLM generate answer
    8. Store answer + thinking in memory
    9. combine_ents every question
    """

    def __init__(
        self,
        word_emb,
        sentence_emb,
        llm: OpenAILLM,
        chunking_api: Optional[str] = None,
        parser_choice: str = "llm",
        top_m: int = 20,
        top_k: int = 200,
        combine_ent_sim: float = 0.93,
        semantic_overlap_sim: float = 0.9,
        answers_choice: str = "overlap",
        thinkings_choice: str = "overlap",
        compress_rate: float = 0.2,
    ):
        self.word_emb = word_emb
        self.sentence_emb = sentence_emb
        self.llm = llm
        self.chunking_api = chunking_api
        self.parser_choice = parser_choice
        self.top_m = top_m
        self.top_k = top_k
        self.combine_ent_sim = combine_ent_sim
        self.semantic_overlap_sim = semantic_overlap_sim
        self.compress_rate = compress_rate
        self.answers_choice = answers_choice
        self.thinkings_choice = thinkings_choice

        # Set extract functions (matches set_includings in original)
        if answers_choice == "overlap":
            self.answers_extract_function = partial(
                get_unique_or_overlap_by_sentence_embedded,
                sim_threshold=semantic_overlap_sim,
            )
        elif answers_choice == "unique":
            self.answers_extract_function = partial(
                get_unique_or_overlap_by_sentence_embedded,
                unique=True, sim_threshold=semantic_overlap_sim,
            )
        else:
            self.answers_extract_function = None

        if thinkings_choice == "overlap":
            self.thinking_extract_function = partial(
                get_unique_or_overlap_by_sentence_embedded,
                sim_threshold=semantic_overlap_sim,
            )
        elif thinkings_choice == "unique":
            self.thinking_extract_function = partial(
                get_unique_or_overlap_by_sentence_embedded,
                unique=True, sim_threshold=semantic_overlap_sim,
            )
        else:
            self.thinking_extract_function = None

        # Whether LLM returns (answer, thinking) tuple
        self.include_thinkings = getattr(llm, 'include_thinkings', False)

        # Memory codebook — stores answers and thinkings from previous Q&A
        self.meta_codebook = {}

        # Stats
        self.questions_processed = 0
        self.total_answers_stored = 0

    # ─────────────────────────────────────────────────────────
    # Step 1: Parse RAG context into temporary facts codebook
    #   Chunks long context first (matching original pipeline's
    #   preload_context_json → chunk_then_subchunk logic)
    #   then passes as list to get_code_book for per-chunk parsing.
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _compute_parse_max_tokens(text: str, base_tokens: int = 256, compress_rate: float = 0.2, cap: int = 4096) -> int:
        """Scale parser max_new_tokens proportionally to input length.

        Formula: base_tokens + input_tokens * compress_rate
        Example: 1000 input tokens, rate 0.2 → 200 + 256 = 456 max output tokens
        """
        input_tokens = len(text or "") // 4  # ~4 chars per token
        scaled = base_tokens + int(input_tokens * compress_rate)
        return min(max(scaled, base_tokens), cap)

    def _parse_context_to_codebook(self, context_text: str) -> dict:
        """Parse RAG's retrieved context into a temporary facts codebook.
        Scales parser output tokens proportionally to input length so longer
        contexts produce more triples (single codebook, no chunking)."""
        if not context_text or not context_text.strip():
            return {}
        try:
            from graph_generator.llm_parser import triplet_parser_llm
            max_tok = self._compute_parse_max_tokens(context_text, compress_rate=self.compress_rate)
            # Call triplet_parser_llm directly with scaled max_new_tokens
            triples = triplet_parser_llm(
                context_text,
                api=self.chunking_api,
                max_new_tokens=max_tok,
            )
            if not triples:
                return {}
            # Build codebook from triples (same as get_code_book single-string path)
            from AutoPrunedRetriever_advanced_cached import (
                build_codebook_from_triples, edges_from_triples,
                all_chains_no_subchains,
            )
            codebook, ent2id, rel2id = build_codebook_from_triples(
                triples, "Store factual statements."
            )
            edges = edges_from_triples(triples, ent2id, rel2id)
            codebook.update({
                "edges([e,r,e])": edges,
                "facts(edges[i])": all_chains_no_subchains(edges, False),
            })
            codebook.pop("sid", None)
            return codebook
        except Exception as e:
            print(f"  [APL] Failed to parse context: {e}")
            return {}

    # ─────────────────────────────────────────────────────────
    # Step 2: Parse question
    # ─────────────────────────────────────────────────────────
    def _parse_question_to_codebook(self, question: str) -> dict:
        """Parse question into KG triples."""
        try:
            return get_code_book(
                question, type="questions", rule="Answer questions.",
                parser_choice=self.parser_choice, api=self.chunking_api,
                sent_emb=self.word_emb,
            )
        except Exception as e:
            print(f"  [APL] Failed to parse question: {e}")
            return {}

    # ─────────────────────────────────────────────────────────
    # Step 3: Merge question into meta_codebook
    # ─────────────────────────────────────────────────────────
    def _merge_question_into_memory(self, question_cb: dict):
        """Merge question codebook into meta_codebook so retrieval can use it."""
        if question_cb and question_cb.get("e"):
            self.meta_codebook = merging_codebook(
                self.meta_codebook, question_cb, "questions", self.word_emb, True
            )

    # ─────────────────────────────────────────────────────────
    # Step 4: Retrieve relevant past answers from memory
    # ─────────────────────────────────────────────────────────
    def _retrieve_from_memory(self) -> Tuple[list, list]:
        """
        Retrieve relevant past answers from meta_codebook.
        Returns (all_answers, all_q_indices).
        """
        if not self.meta_codebook or not self.meta_codebook.get("answers_lst"):
            return [], []

        questions_lst = self.meta_codebook.get("questions_lst", [])
        if not questions_lst:
            return [], []

        questions_edges_index = questions_lst[-1]
        _answers_lst = self.meta_codebook.get("answers_lst", [])

        if not _answers_lst:
            return [], []

        try:
            adapted_m = min(max(1, int(0.1 * len(_answers_lst))), self.top_m)
            # Match original: minimum floor for retrieval budget
            adapted_m = max(adapted_m, min(5, self.top_m))

            top_m_results = coarse_filter_torch(
                questions=questions_edges_index,
                codebook_main=self.meta_codebook,
                sentence_emb=self.sentence_emb,
                top_k=self.top_k,
                top_m=adapted_m,
                target='questions',
            )
            all_answers, all_q_indices = get_all_results_entire_chunk(
                top_m_results, self.meta_codebook, 'answers'
            )
            return all_answers, all_q_indices
        except Exception as e:
            print(f"  [APL] Retrieval from memory failed: {e}")
            return [], []

    # ─────────────────────────────────────────────────────────
    # Step 5: find_related_knowledge
    #   Matches original: applies answers_extract_function
    #   (get_unique_or_overlap_by_sentence_embedded) to filter/dedup
    #   retrieved answers before passing to compact_indicies_for_prompt.
    # ─────────────────────────────────────────────────────────
    def _find_related_knowledge(self, all_answers, all_q_indices):
        """
        Filter/dedup retrieved answers + get thinkings.
        Returns domain_knowledge_lst = [final_answers_lsts, final_thinkings_lsts]
        matching original find_related_knowledge.
        """
        domain_knowledge_lst = []

        # Answers
        if all_answers:
            flat_answers = get_flat_answers_lsts(all_answers)
            if flat_answers and self.answers_extract_function:
                final_answers_lsts = self.answers_extract_function(
                    self.meta_codebook, flat_answers, self.sentence_emb
                )
                if not final_answers_lsts:
                    final_answers_lsts = flat_answers
            else:
                final_answers_lsts = flat_answers if flat_answers else []
            domain_knowledge_lst.append(final_answers_lsts)
        else:
            domain_knowledge_lst.append([])

        # Thinkings
        if self.include_thinkings:
            if all_q_indices and self.meta_codebook.get("thinkings_lst"):
                thinkings_lsts = get_thinkings_lsts(all_q_indices, self.meta_codebook)
                flat_thinkings = get_flat_answers_lsts(thinkings_lsts)
                if flat_thinkings and self.thinking_extract_function:
                    final_thinkings_lsts = self.thinking_extract_function(
                        self.meta_codebook, flat_thinkings, self.sentence_emb
                    )
                    if not final_thinkings_lsts:
                        final_thinkings_lsts = flat_thinkings
                else:
                    final_thinkings_lsts = flat_thinkings if flat_thinkings else []
                domain_knowledge_lst.append(final_thinkings_lsts)
            else:
                domain_knowledge_lst.append([])

        return domain_knowledge_lst

    # ─────────────────────────────────────────────────────────
    # Step 6: compact_indicies_for_prompt — Build final merged JSON
    #   Matches original: uses get_json_with_given_knowledge (+ thinkings)
    #   then merges RAG facts on top.
    # ─────────────────────────────────────────────────────────
    def _build_final_json(
        self, question_cb: dict, facts_cb: dict,
        domain_knowledge_lst: list,
    ) -> dict:
        """
        Build the final merged JSON for LLM prompt.
        Matches original compact_indicies_for_prompt logic.
        """
        flat_answers_lsts = domain_knowledge_lst[0] if len(domain_knowledge_lst) > 0 else None
        flat_thinkings_lsts = None
        if self.include_thinkings and len(domain_knowledge_lst) > 1:
            flat_thinkings_lsts = domain_knowledge_lst[1]

        # Match original compact_indicies_for_prompt branching
        if flat_answers_lsts and flat_thinkings_lsts and self.include_thinkings:
            try:
                final = get_json_with_given_knowledge_and_thinkings(
                    flat_answers_lsts, flat_thinkings_lsts,
                    self.meta_codebook, question_cb,
                )
            except Exception as e:
                print(f"  [APL] get_json_with_given_knowledge_and_thinkings failed: {e}")
                final = self._fallback_question_only(question_cb)
        elif flat_answers_lsts:
            try:
                final = get_json_with_given_knowledge(
                    flat_answers_lsts, self.meta_codebook, question_cb,
                )
            except Exception as e:
                print(f"  [APL] get_json_with_given_knowledge failed: {e}")
                final = self._fallback_question_only(question_cb)
        else:
            final = self._fallback_question_only(question_cb)

        # Merge RAG's temporary facts (same as original facts section)
        if facts_cb and facts_cb.get("e"):
            final = self._merge_facts_into_final(final, facts_cb)

        return final

    def _fallback_question_only(self, question_cb: dict) -> dict:
        """When no memory answers available, just format question codebook."""
        if not question_cb:
            return {}
        final = copy.deepcopy(question_cb)
        # Normalize key name (matches original else branch)
        if "edges([e,r,e])" in final and "edge_matrix" not in final:
            final["edge_matrix"] = final.pop("edges([e,r,e])")
        if "questions(edges[i])" in final:
            final["questions([[e,r,e], ...])"] = decode_questions(
                final["questions(edges[i])"], final, "edges"
            )
        return final

    def _merge_facts_into_final(self, final: dict, facts_cb: dict) -> dict:
        """
        Merge RAG's facts into the final JSON.
        Copies from compact_indicies_for_prompt's facts merging logic:
        remaps entity/relation indices from facts_cb into final's space.
        """
        facts_edges_key = "facts(edges[i])"
        raw_facts_runs = facts_cb.get(facts_edges_key, [])
        facts_edge_matrix = facts_cb.get("edges([e,r,e])", facts_cb.get("edge_matrix", []))

        if not raw_facts_runs or not facts_edge_matrix:
            return final

        E_final = final.get("e", [])
        R_final = final.get("r", [])
        em_final = final.get("edge_matrix", [])

        e_name2idx = {name: i for i, name in enumerate(E_final)}
        r_name2idx = {name: i for i, name in enumerate(R_final)}
        tuple2idx = {tuple(e): i for i, e in enumerate(em_final)}

        facts_E = facts_cb.get("e", [])
        facts_R = facts_cb.get("r", [])

        def ensure_ent(old_idx):
            name = facts_E[old_idx]
            idx = e_name2idx.get(name)
            if idx is None:
                idx = len(E_final)
                E_final.append(name)
                e_name2idx[name] = idx
            return idx

        def ensure_rel(old_idx):
            name = facts_R[old_idx]
            idx = r_name2idx.get(name)
            if idx is None:
                idx = len(R_final)
                R_final.append(name)
                r_name2idx[name] = idx
            return idx

        def ensure_edge(old_edge_idx):
            e1, r, e2 = facts_edge_matrix[old_edge_idx]
            h = ensure_ent(e1)
            rel = ensure_rel(r)
            t = ensure_ent(e2)
            tup = (h, rel, t)
            idx = tuple2idx.get(tup)
            if idx is None:
                idx = len(em_final)
                em_final.append([h, rel, t])
                tuple2idx[tup] = idx
            return idx

        remapped_facts = [[ensure_edge(i) for i in run] for run in raw_facts_runs]

        final["e"] = E_final
        final["r"] = R_final
        final["edge_matrix"] = em_final
        final["facts(edges[i])"] = remapped_facts
        final["facts([[e,r,e], ...])"] = decode_questions(remapped_facts, final, "edges")

        return final

    # ─────────────────────────────────────────────────────────
    # Step 7: Format and generate answer
    # ─────────────────────────────────────────────────────────
    def _generate_answer(self, final_json: dict, question: str) -> Tuple[str, str, Optional[str]]:
        """
        Generate answer using properly formatted structured context.
        Returns (answer, fact_context, thinking_or_None).
        """
        try:
            # Select best context (readable text for logging)
            q_txt, gk_txt, st_txt, ft_txt = select_best_context_by_keys(final_json)
            fact_context = ""
            if ft_txt:
                fact_context += ft_txt
            if gk_txt:
                fact_context += gk_txt

            # Slice and format the JSON for LLM prompt
            sliced = slice_for_final_merged_json(final_json, use_word_format=True)

            # Generate answer
            result = self.llm.take_questions(
                sliced, question, retrieval_time=0.0
            )
            if isinstance(result, tuple):
                return result[0], fact_context, result[1]
            return result, fact_context, None
        except Exception as e:
            print(f"  [APL] Generation failed: {e}")
            return "I don't know.", "", None

    # ─────────────────────────────────────────────────────────
    # Step 8: Store answer (+ thinking) in memory
    #   Matches original update_meta: parses answer and thinking,
    #   merges both into meta_codebook.
    # ─────────────────────────────────────────────────────────
    def _store_answer_in_memory(self, answer: str, thinking: Optional[str] = None):
        """Store answer and thinking in meta_codebook. Matches original update_meta."""
        # Parse answer
        answer_cb = None
        if answer and not is_no_answer_text(answer):
            try:
                answer_cb = get_code_book(
                    answer, type="answers", rule="Store answer information.",
                    parser_choice=self.parser_choice, api=self.chunking_api,
                    sent_emb=self.word_emb,
                )
            except Exception as e:
                print(f"  [APL] Failed to parse answer: {e}")

        # Parse thinking
        thinking_cb = None
        if self.include_thinkings and thinking:
            try:
                thinking_cb = get_code_book(
                    thinking, type="thinkings", rule="Store thinking process.",
                    parser_choice=self.parser_choice, api=self.chunking_api,
                    sent_emb=self.word_emb,
                )
            except Exception as e:
                print(f"  [APL] Failed to parse thinking: {e}")

        # Merge into meta_codebook (matches original update_meta logic)
        if answer_cb and answer_cb.get("e") and len(answer_cb.get("edges([e,r,e])", [])) > 0:
            self.meta_codebook = merging_codebook(
                self.meta_codebook, answer_cb, "answers", self.word_emb, True
            )
            self.total_answers_stored += 1

            # Store thinking only if answer was valid
            if self.include_thinkings and thinking_cb and thinking_cb.get("e"):
                self.meta_codebook = merging_codebook(
                    self.meta_codebook, thinking_cb, "thinkings", self.word_emb, True
                )
        else:
            # Rollback: pop the question we just merged since answer was empty
            if self.meta_codebook.get("questions_lst"):
                self.meta_codebook["questions_lst"].pop()
                if self.meta_codebook.get("questions_lst_embedding"):
                    self.meta_codebook["questions_lst_embedding"].pop()
                if self.meta_codebook.get("questions_compressed_lst"):
                    self.meta_codebook["questions_compressed_lst"].pop()
                if self.meta_codebook.get("questions_groups_lst"):
                    self.meta_codebook["questions_groups_lst"].pop()

    # ─────────────────────────────────────────────────────────
    # Step 9: Combine entities (dedup) — every question
    # ─────────────────────────────────────────────────────────
    def _combine_ents(self):
        """Run entity deduplication on meta_codebook."""
        if not _HAS_COMBINE:
            return
        if not self.meta_codebook or len(self.meta_codebook.get("e", [])) < 2:
            return
        try:
            self.meta_codebook = combine_ents_ann_knn(
                self.meta_codebook,
                sim_threshold=self.combine_ent_sim,
                word_emb=self.word_emb,
            )
        except Exception as e:
            print(f"  [APL] combine_ents failed: {e}")

    # ─────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────
    def process_question(
        self, question: str, rag_context: str, store_answer: bool = True,
    ) -> Tuple[str, str]:
        """
        Full pipeline for one question (matches original run_work_flow):
        1. Parse RAG context → temporary facts KG
        2. Parse question → question KG
        3. Merge question into meta_codebook
        4. Retrieve answers from memory (coarse_filter_torch)
        5. find_related_knowledge — filter/dedup retrieved answers
        6. compact_indicies_for_prompt — build final JSON
        7. slice_for_final_merged_json + generate answer
        8. Store answer + thinking in memory
        9. combine_ents every question
        """
        self.questions_processed += 1

        # 1. Parse RAG's context into temporary facts codebook
        facts_cb = self._parse_context_to_codebook(rag_context)

        # 2. Parse question
        question_cb = self._parse_question_to_codebook(question)

        # 3. Merge question into meta_codebook
        self._merge_question_into_memory(question_cb)

        # 4. Retrieve relevant past answers from memory
        all_answers, all_q_indices = self._retrieve_from_memory()

        # 5. find_related_knowledge — filter/dedup answers + get thinkings
        domain_knowledge_lst = self._find_related_knowledge(all_answers, all_q_indices)

        # 6. Build final JSON (compact_indicies_for_prompt + merge RAG facts)
        final_json = self._build_final_json(
            question_cb, facts_cb, domain_knowledge_lst
        )

        # 7. Generate answer with proper formatting
        answer, fact_context, thinking = self._generate_answer(final_json, question)

        # 8. Store answer + thinking in memory
        if store_answer:
            self._store_answer_in_memory(answer, thinking)

        # 9. combine_ents every question (matches original pipeline)
        self._combine_ents()

        return answer, fact_context

    def get_stats(self) -> dict:
        return {
            "questions_processed": self.questions_processed,
            "answers_in_memory": self.total_answers_stored,
            "codebook_entities": len(self.meta_codebook.get("e", [])),
            "codebook_relations": len(self.meta_codebook.get("r", [])),
            "codebook_edges": len(self.meta_codebook.get("edge_matrix", [])),
        }


# ═════════════════════════════════════════════════════════════
# CLI runner
# ═════════════════════════════════════════════════════════════

def _get_rag_context(pred: dict) -> str:
    """Extract RAG retrieved context only (not ground-truth evidence)."""
    return pred.get("context", "")


def run_apl_on_predictions(
    predictions_path: str,
    output_path: str,
    llm_api: Optional[str] = None,
    skip_update_meta: bool = False,
    max_questions: Optional[int] = None,
):
    """Run AutoPruned Layer on a baseline RAG's predictions file."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("=" * 60)
    print("AutoPruned Layer (APL) v3")
    print("  - find_related_knowledge (answer overlap/dedup)")
    print("  - get_json_with_given_knowledge (proper index remap)")
    print("  - thinking stored in memory")
    print("  - compact facts merging")
    print("  - slice_for_final_merged_json")
    print("  - combine_ents every question")
    print("=" * 60)

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if max_questions:
        predictions = predictions[:max_questions]
    print(f"  {len(predictions)} questions to process")

    print("Initializing embeddings and LLM...")
    word_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    api_key = llm_api or os.environ.get("OPENAI_API_KEY")
    llm = OpenAILLM(
        include_thinkings=True,
        model_name="gpt-4o-mini",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        api_key=api_key,
    )

    apl = AutoPrunedLayer(
        word_emb=word_emb,
        sentence_emb=word_emb,
        llm=llm,
        chunking_api=api_key,
        parser_choice="llm",
    )

    print(f"\nProcessing {len(predictions)} questions...")
    results = []
    start_time = time.time()

    for i, pred in enumerate(predictions):
        question = pred.get("question", "")
        rag_context = _get_rag_context(pred)
        ground_truth = pred.get("ground_truth", pred.get("answer", ""))

        print(f"\n[{i+1}/{len(predictions)}] {question[:60]}...")

        t0 = time.time()
        answer, fact_context = apl.process_question(
            question=question,
            rag_context=rag_context,
            store_answer=not skip_update_meta,
        )
        latency = time.time() - t0

        result = {
            "id": pred.get("id"),
            "question": question,
            "source": pred.get("source"),
            "context": fact_context or "",
            "evidence": pred.get("evidence", []),
            "question_type": pred.get("question_type"),
            "generated_answer": answer,
            "ground_truth": ground_truth,
            "original_answer": pred.get("generated_answer", ""),
            "original_context": rag_context,
            "latency_sec": latency,
        }
        results.append(result)

        stats = apl.get_stats()
        print(f"  Answer: {answer[:80]}...")
        print(f"  Memory: {stats['answers_in_memory']} answers, "
              f"{stats['codebook_entities']} entities")
        print(f"  Context: {len(fact_context or '')} chars")

        if (i + 1) % 50 == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=True)
            print(f"  [Checkpoint] Saved {len(results)} results")

    total_time = (time.time() - start_time) / 60
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(f"\n{'='*60}")
    print(f"APL v3 Complete")
    print(f"  Questions: {len(results)}")
    print(f"  Time: {total_time:.1f} min")
    print(f"  Final memory: {apl.get_stats()}")
    print(f"  Output: {output_path}")
    print(f"  Token stats: {dict(TOKEN_STATS)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoPruned Layer v3")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm_api", default=None)
    parser.add_argument("--skip_update_meta", action="store_true")
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    run_apl_on_predictions(
        predictions_path=args.predictions,
        output_path=args.output,
        llm_api=args.llm_api,
        skip_update_meta=args.skip_update_meta,
        max_questions=args.max_questions,
    )
