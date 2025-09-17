def _all_contiguous_subseqs(seq, min_len=2):
    """Generate all contiguous subsequences of seq with length >= min_len."""
    n = len(seq)
    for i in range(n):
        for j in range(i + min_len, n + 1):
            yield tuple(seq[i:j])

def _is_subrun(a, b):
    """Check if subsequence a is fully inside b (both are tuples)."""
    return len(a) < len(b) and any(
        b[i:i+len(a)] == a for i in range(len(b) - len(a) + 1)
    )

def common_contiguous_overlaps_naive(answers_lst, min_len=2):
    """
    Naïve version: find all maximal common contiguous subsequences.
    """
    if not answers_lst:
        return []

    candidates = set(_all_contiguous_subseqs(answers_lst[0], min_len=min_len))

    for lst in answers_lst[1:]:
        runs_here = set(_all_contiguous_subseqs(lst, min_len=min_len))
        candidates &= runs_here
        if not candidates:
            return []

    maximal = set(candidates)
    for a in list(candidates):
        for b in candidates:
            if a != b and _is_subrun(a, b):
                maximal.discard(a)
                break

    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]

# Rolling-hash-based approach
def common_contiguous_overlaps_hash(lists, min_len=2):
    """
    Corrected hash-based implementation that matches naive semantics.
    Finds maximal common contiguous subsequences in all lists.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return [lists[0]]

    # Collect all candidates from the first list
    first = lists[0]
    candidates = set()
    for i in range(len(first)):
        for j in range(i + min_len, len(first) + 1):
            candidates.add(tuple(first[i:j]))

    # Intersect with candidates from the rest
    for lst in lists[1:]:
        runs_here = set()
        for i in range(len(lst)):
            for j in range(i + min_len, len(lst) + 1):
                runs_here.add(tuple(lst[i:j]))
        candidates &= runs_here
        if not candidates:
            return []

    # Keep only maximal runs
    maximal = set(candidates)
    for a in list(candidates):
        for b in candidates:
            if a != b and len(a) < len(b) and any(
                tuple(b[i:i+len(a)]) == a for i in range(len(b) - len(a) + 1)
            ):
                maximal.discard(a)
                break

    # Sort by length desc, then lex for determinism
    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]

from collections import defaultdict

def common_contiguous_overlaps_advanced(lists, min_len=2, min_support_ratio=0.7):
    """
    Find maximal contiguous subsequences that appear in at least `min_support` lists.
    """
    if not lists:
        return []

    min_support = max(1, round(len(lists) * min_support_ratio))
    print("min_support: ", min_support)

    # Collect all candidates from all lists
    candidate_counts = defaultdict(set)  # subseq -> set of list indices
    for idx, lst in enumerate(lists):
        for i in range(len(lst)):
            for j in range(i + min_len, len(lst) + 1):
                subseq = tuple(lst[i:j])
                candidate_counts[subseq].add(idx)

    # Keep only those with enough support
    frequent = {subseq for subseq, idxs in candidate_counts.items() if len(idxs) >= min_support}

    # Filter to maximal
    maximal = set(frequent)
    for a in list(frequent):
        for b in frequent:
            if a != b and len(a) < len(b):
                if any(tuple(b[i:i+len(a)]) == a for i in range(len(b)-len(a)+1)):
                    maximal.discard(a)
                    break

    # Return sorted list
    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]



# ----------------- TEST MAIN -----------------
if __name__ == "__main__":
    import random

    # If you want exactly the numbers 1-100 in random order
    # All lists can draw from the same range (1-300)
    # A1 = random.sample(range(1, 3001), 1000)  # 100 numbers from 1-300
    # A2 = random.sample(range(1, 3001), 1000)  # 100 numbers from 1-300
    # A3 = random.sample(range(1, 3001), 1000)  # 100 numbers from 1-300
    #
    # # Check for overlap
    # all_numbers = A1 + A2 + A3
    # unique_numbers = set(all_numbers)
    # overlap_count = len(all_numbers) - len(unique_numbers)

    # print(f"Total numbers across lists: {len(all_numbers)}")
    # print(f"Unique numbers: {len(unique_numbers)}")
    # print(f"Overlap (duplicates): {overlap_count}")

    answers = [
        ["A", "B", "C", "D", "E"],  # 主序列
        ["X", "B", "C", "D", "Y"],  # 和 1 有 [B,C,D]
        ["B", "C", "D", "Z"],  # 和 1、2 有 [B,C,D]
        ["A", "B", "C", "Q", "R"],  # 和 1 有 [A,B,C]
        ["M", "N", "O", "P"],  # 完全无 overlap
        ["C", "D", "E", "F", "G"],  # 和 1 有 [C,D,E]
    ]

    import timeit
    start = timeit.timeit()
    print("Naive:", common_contiguous_overlaps_naive(answers, min_len=1))
    end = timeit.timeit()
    print(end - start)

    start = timeit.timeit()
    print("Hash :", common_contiguous_overlaps_hash(answers, min_len=1))
    end = timeit.timeit()
    print(end - start)

    print("Hash :", common_contiguous_overlaps_advanced(answers, min_len=1))