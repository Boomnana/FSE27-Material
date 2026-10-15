from __future__ import annotations

from collections import Counter
from fractions import Fraction


def _labels(partition):
    return {x: c for c, block in enumerate(partition) for x in block}


def score_partition(predicted, truth):
    pred = _labels(predicted)
    gold = _labels(truth)
    ids = sorted(gold)
    if not ids or not set(ids) <= set(pred):
        raise ValueError("truth scope must be nonempty and contained in prediction")
    pc = Counter(pred[i] for i in ids)
    gc = Counter(gold[i] for i in ids)
    cells = Counter((pred[i], gold[i]) for i in ids)
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(v) for v in cells.values())
    pp = sum(c2(v) for v in pc.values())
    gp = sum(c2(v) for v in gc.values())
    fp, fn = pp - tp, gp - tp
    total = c2(len(ids))
    expected = Fraction(pp * gp, total) if total else Fraction(0)
    maximum = Fraction(pp + gp, 2)
    ari = Fraction(1) if maximum == expected else (tp - expected) / (maximum - expected)
    bp = sum((Fraction(v * v, pc[p]) for (p, g), v in cells.items()), Fraction()) / len(ids)
    br = sum((Fraction(v * v, gc[g]) for (p, g), v in cells.items()), Fraction()) / len(ids)
    bf = Fraction(0) if bp + br == 0 else 2 * bp * br / (bp + br)
    return {"reports": len(ids), "pair_tp": tp, "pair_fp": fp, "pair_fn": fn,
            "pair_errors": fp + fn, "ari": float(ari),
            "b3_precision": float(bp), "b3_recall": float(br), "b3_f1": float(bf)}


def partition_from_issue_map(report_ids, issue_by_id):
    groups = {}
    for i, rid in enumerate(report_ids):
        if rid in issue_by_id:
            groups.setdefault(issue_by_id[rid], []).append(i)
    return tuple(tuple(v) for _, v in sorted(groups.items(), key=lambda kv: kv[0]))
