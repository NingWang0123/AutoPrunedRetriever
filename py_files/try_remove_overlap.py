from typing import List, Dict

from py_files.try_overlap import common_contiguous_overlaps_advanced


# --- Overlap Finder with min_support_ratio ---
# def _all_contiguous_subseqs(seq, min_len=2):
#     """Generate all contiguous subsequences of a list with length >= min_len."""
#     n = len(seq)
#     for i in range(n):
#         for j in range(i + min_len, n + 1):
#             yield tuple(seq[i:j])
#
# def common_overlaps_min_support(lists, min_len=2, min_support_ratio=1.0):
#     """
#     Find common contiguous subsequences across multiple lists.
#     Appear in at least `round(len(lists) * min_support_ratio)` lists.
#     """
#     if not lists:
#         return []
#
#     min_support = max(1, round(len(lists) * min_support_ratio))
#
#     # collect subsequence → set of list_ids where it appears
#     subseq_occurrences: Dict[tuple, set] = {}
#     for idx, seq in enumerate(lists):
#         seen_here = set()
#         for subseq in _all_contiguous_subseqs(seq, min_len=min_len):
#             if subseq not in seen_here:  # avoid double counting same seq
#                 subseq_occurrences.setdefault(subseq, set()).add(idx)
#                 seen_here.add(subseq)
#
#     # keep those with enough support
#     candidates = [subseq for subseq, occs in subseq_occurrences.items()
#                   if len(occs) >= min_support]
#
#     # prune subruns
#     maximal = set(candidates)
#     for a in list(candidates):
#         for b in candidates:
#             if a != b and len(a) < len(b):
#                 for i in range(len(b) - len(a) + 1):
#                     if b[i:i+len(a)] == a:
#                         maximal.discard(a)
#                         break
#
#     return [list(s) for s in sorted(maximal, key=lambda t: (-len(t), t))]

def get_unique_knowledge_naive(overlapped_answers,flat_answers_lsts):
    """
    For each overlap run in overlapped_answers['overlaps'], keep that run only
    in the 'owner' sequence (the one with the longest continuation after the run),
    and remove the run from all other sequences where it appears.

    Inputs:
    overlapped_answers: {'overlaps': [[edges_index, edges_index,...],...]}

    flat_answers_lsts: [[edges_index,...],...] ; get from get_flat_answers_lsts(answers_lsts)
    """

    # Normalize inputs
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    def find_run_positions(run: List[int], seq: List[int]) -> List[int]:
        L = len(run)
        if L == 0 or L > len(seq):
            return []
        return [i for i in range(len(seq) - L + 1) if seq[i:i + L] == run]

    def remove_all_runs(seq: List[int], run: List[int]) -> List[int]:
        """Remove all non-overlapping occurrences of run from seq (greedy left-to-right)."""
        res: List[int] = []
        i = 0
        L = len(run)
        n = len(seq)
        while i <= n - L:
            if seq[i:i+L] == run:
                i += L  # skip the run
            else:
                res.append(seq[i])
                i += 1
        # append trailing tail
        res.extend(seq[i:])
        return res

    # Process longer overlaps first to avoid smaller runs interfering
    runs_sorted = sorted(runs, key=len, reverse=True)

    assignments = []
    for run in runs_sorted:
        # Find occurrences in each sequence
        occs: Dict[int, List[int]] = {idx: find_run_positions(run, seq) for idx, seq in enumerate(out_answers)}
        present = {i: pos for i, pos in occs.items() if pos}
        if not present:
            continue  # this run doesn't appear anywhere

        # Choose owner: sequence with the maximum tail length after the best occurrence
        L = len(run)
        owner = None
        best_tail = -1
        best_total_len = -1
        for i, positions in present.items():
            for pos in positions:
                tail_len = len(out_answers[i]) - (pos + L)
                # Tie-breakers: longer total sequence length, then smaller index
                if (tail_len > best_tail or
                    (tail_len == best_tail and len(out_answers[i]) > best_total_len) or
                    (tail_len == best_tail and len(out_answers[i]) == best_total_len and (owner is None or i < owner))):
                    owner = i
                    best_tail = tail_len
                    best_total_len = len(out_answers[i])

        # Remove this run from all non-owner sequences where it occurs
        for j in range(len(out_answers)):
            if j != owner and occs.get(j):
                out_answers[j] = remove_all_runs(out_answers[j], run)

        assignments.append({
            'run': run,
            'owner': owner,
            'occurrences': {i: occs[i] for i in present}
        })


    return {'assignments': assignments, 'out_answers': out_answers}

# --- Optimized Unique Knowledge ---
def get_unique_knowledge_efficient(overlapped_answers, flat_answers_lsts):
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    runs_sorted = sorted(runs, key=len, reverse=True)
    assignments = []

    for run in runs_sorted:
        L = len(run)
        if L == 0: continue

        # Find occurrences
        occs = {}
        for idx, seq in enumerate(out_answers):
            positions = [i for i in range(len(seq) - L + 1) if seq[i:i+L] == run]
            if positions:
                occs[idx] = positions

        if not occs: continue

        # Pick owner
        owner, best_tail, best_len = None, -1, -1
        for i, positions in occs.items():
            for pos in positions:
                tail_len = len(out_answers[i]) - (pos + L)
                if (tail_len > best_tail or
                   (tail_len == best_tail and len(out_answers[i]) > best_len) or
                   (tail_len == best_tail and len(out_answers[i]) == best_len and (owner is None or i < owner))):
                    owner, best_tail, best_len = i, tail_len, len(out_answers[i])

        # Remove from non-owners
        for j in occs:
            if j != owner:
                new_seq, skip = [], 0
                seq = out_answers[j]
                for k in range(len(seq)):
                    if skip: skip -= 1; continue
                    if seq[k:k+L] == run:
                        skip = L - 1
                        continue
                    new_seq.append(seq[k])
                out_answers[j] = new_seq

        assignments.append({"run": run, "owner": owner, "occurrences": occs})

    return {"assignments": assignments, "out_answers": out_answers}

from typing import List, Dict, Any

def get_unique_knowledge_advanced(overlapped_answers, flat_answers_lsts,
                                  alpha=1.0, beta=0.5, gamma=0.5):
    """
    Smarter version: assign each overlap run to exactly ONE owner sequence
    using a weighted scoring function.
    score=α⋅tail_len + β⋅seq_len + γ⋅frequency (frequency: how many times this run appears in the sequence)
    high tail_len -> more following description
    high seq_len -> more general information
    high frequency -> likely closer description of the overlapped_answers
    """
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    def find_run_positions(run: List[int], seq: List[int]) -> List[int]:
        L = len(run)
        if L == 0 or L > len(seq):
            return []
        return [i for i in range(len(seq) - L + 1) if seq[i:i + L] == run]

    def remove_all_runs(seq: List[int], run: List[int]) -> List[int]:
        """Remove all non-overlapping occurrences of run from seq (greedy left-to-right)."""
        res, i, L, n = [], 0, len(run), len(seq)
        while i <= n - L:
            if seq[i:i+L] == run:
                i += L
            else:
                res.append(seq[i]); i += 1
        res.extend(seq[i:])
        return res

    runs_sorted = sorted(runs, key=len, reverse=True)
    assignments = []

    for run in runs_sorted:
        occs: Dict[int, List[int]] = {idx: find_run_positions(run, seq) for idx, seq in enumerate(out_answers)}
        present = {i: pos for i, pos in occs.items() if pos}
        if not present:
            continue

        # --- scoring ---
        best_score, owner = -1e9, None
        for i, positions in present.items():
            seq = out_answers[i]
            seq_len = len(seq)
            freq = len(positions)
            for pos in positions:
                tail_len = seq_len - (pos + len(run))
                score = alpha * tail_len + beta * seq_len + gamma * freq
                if score > best_score:
                    best_score, owner = score, i

        # remove from all non-owner
        for j in range(len(out_answers)):
            if j != owner and occs.get(j):
                out_answers[j] = remove_all_runs(out_answers[j], run)

        assignments.append({'run': run, 'owner': owner, 'score': best_score})

    return {'assignments': assignments, 'out_answers': out_answers}


# ---------------- TESTS ----------------
if __name__ == "__main__":
    A1 = [1, 2, 3, 4, 5, 6]
    A2 = [1, 2, 3, 4, 2, 3, 4, 9, 2, 3, 4, 10]
    A3 = [11, 2, 3, 7, 4, 8]

    lists = [A1, A2, A3]

    print("=== Overlap with min_support_ratio=1.0 (must appear in all lists) ===")
    overlaps = common_contiguous_overlaps_advanced(lists, min_len=2, min_support_ratio=0.66)
    print(overlaps)  # expect [[2,3,4]]
    #
    # print("\n=== Overlap with min_support_ratio=0.66 (appear in ≥2 of 3 lists) ===")
    # overlaps = common_overlaps_min_support(lists, min_len=2, min_support_ratio=0.66)
    # print(overlaps)  # expect [[2,3,4]]

    print("\n=== Unique Knowledge Assignment ===")
    overlapped_answers = {"overlaps": [[2,3,4]]}
    result = get_unique_knowledge_efficient(overlapped_answers, lists)
    print("Efficient: ", result)

    result = get_unique_knowledge_naive(overlapped_answers, lists)
    print("Naive: ", result)

    result = get_unique_knowledge_advanced(overlapped_answers, lists)
    print("Advanced: ", result)
