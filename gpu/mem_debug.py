"""
Memory diagnostic for the RAG pipeline.
Import and call snapshot() at key points to track where RAM goes.

Usage:
    from mem_debug import snapshot, report
    
    snapshot("after model load")
    snapshot("after codebook load")
    snapshot("before retrieve_new")
    snapshot("after retrieve_new")
    report()
"""
import psutil
import os
import gc
import sys

_snapshots = []

def snapshot(label=""):
    """Record current process memory usage."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    rss_gb = mem.rss / 1e9
    vms_gb = mem.vms / 1e9
    
    sys_mem = psutil.virtual_memory()
    sys_free_gb = sys_mem.available / 1e9
    
    _snapshots.append({
        'label': label,
        'rss_gb': rss_gb,
        'vms_gb': vms_gb,
        'sys_free_gb': sys_free_gb,
    })
    print(f"[MEM] {label:40s} | RSS={rss_gb:.2f}GB | VMS={vms_gb:.2f}GB | SysFree={sys_free_gb:.1f}GB")
    return rss_gb


def report():
    """Print all snapshots."""
    print("\n" + "=" * 80)
    print("MEMORY REPORT")
    print("=" * 80)
    prev_rss = 0
    for s in _snapshots:
        delta = s['rss_gb'] - prev_rss
        print(f"  {s['label']:40s} | RSS={s['rss_gb']:.2f}GB | Δ={delta:+.2f}GB | SysFree={s['sys_free_gb']:.1f}GB")
        prev_rss = s['rss_gb']
    print("=" * 80)


def codebook_mem_breakdown(codebook, label="codebook"):
    """Show memory usage of each codebook field."""
    print(f"\n[MEM] {label} breakdown:")
    total = 0
    for key, val in codebook.items():
        if val is None:
            continue
        size = _est_size(val)
        total += size
        if size > 1e6:  # only show > 1MB
            print(f"  {key:35s} : {size/1e6:.1f} MB")
    print(f"  {'TOTAL':35s} : {total/1e6:.1f} MB")
    return total


def _est_size(obj, depth=0):
    """Rough memory estimate for common types."""
    if depth > 5:
        return sys.getsizeof(obj)
    
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.nbytes + 112  # numpy overhead
    
    if isinstance(obj, list):
        total = sys.getsizeof(obj)  # list container
        for item in obj:
            total += _est_size(item, depth + 1)
        return total
    
    if isinstance(obj, dict):
        total = sys.getsizeof(obj)
        for k, v in obj.items():
            total += _est_size(k, depth + 1) + _est_size(v, depth + 1)
        return total
    
    return sys.getsizeof(obj)


def force_gc_and_report(label=""):
    """Force garbage collection and report freed memory."""
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss / 1e9
    gc.collect()
    after = proc.memory_info().rss / 1e9
    freed = before - after
    print(f"[GC] {label}: before={before:.2f}GB after={after:.2f}GB freed={freed:.3f}GB")
    return freed
