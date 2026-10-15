from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
import unicodedata
import jieba
import jieba.posseg as pseg
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
tuning = sys.modules[__name__]
def score(partition: list[list[str]], gold: dict[str, str]) -> dict:
    pred = {str(rid): cluster for cluster, group in enumerate(partition) for rid in group}
    if set(pred) != set(gold):
        raise ValueError(
            f"partition/gold population mismatch: missing={len(set(gold)-set(pred))}, "
            f"extra={len(set(pred)-set(gold))}"
        )
    ids = sorted(gold)
    truth = [gold[rid] for rid in ids]
    labels = [pred[rid] for rid in ids]
    pc, gc, cells = Counter(labels), Counter(truth), Counter(zip(labels, truth))
    choose2 = lambda n: n * (n - 1) // 2
    tp = sum(choose2(value) for value in cells.values())
    pp = sum(choose2(value) for value in pc.values())
    gp = sum(choose2(value) for value in gc.values())
    fp, fn = pp - tp, gp - tp
    pair_p = tp / pp if pp else 0.0
    pair_r = tp / gp if gp else 0.0
    b3_p = sum(value * value / pc[p] for (p, _), value in cells.items()) / len(ids)
    b3_r = sum(value * value / gc[g] for (_, g), value in cells.items()) / len(ids)
    return {
        "ari": adjusted_rand_score(truth, labels),
        "pair_precision": pair_p,
        "pair_recall": pair_r,
        "pair_f1": 2 * pair_p * pair_r / (pair_p + pair_r) if pair_p + pair_r else 0.0,
        "b3_precision": b3_p,
        "b3_recall": b3_r,
        "b3_f1": 2 * b3_p * b3_r / (b3_p + b3_r) if b3_p + b3_r else 0.0,
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "clusters": len(pc),
    }


def stable_partition(labels, ids: list[str]) -> list[list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for label, rid in zip(labels, ids):
        groups[int(label)].append(str(rid))
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: (group[0], len(group), group),
    )


def coassociation(partitions: list[list[list[str]]], ids: list[str]) -> np.ndarray:
    index = {rid: i for i, rid in enumerate(ids)}
    votes = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for partition in partitions:
        for group in partition:
            positions = [index[str(rid)] for rid in group]
            votes[np.ix_(positions, positions)] += 1.0
    votes /= len(partitions)
    np.fill_diagonal(votes, 1.0)
    return votes


def tree_from_similarity(similarity: np.ndarray, linkage_name: str):
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method=linkage_name)


def tokenize_words(text: str) -> list[str]:
    return [
        token.lower()
        for token in jieba.lcut(text)
        if re.search(r"[\u3400-\u9fffA-Za-z0-9]", token)
    ]


def token_sets(records: list[dict], kind: str, baseline_module, stopwords: set[str]) -> list[set[str]]:
    result = []
    for row in records:
        text = row["description"]
        clean = re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", text.lower())
        if kind == "pos":
            tokens = baseline_module.ctras_tokens(text, stopwords)
        elif kind == "word":
            tokens = tokenize_words(text)
        elif kind == "char2":
            tokens = [clean[i : i + 2] for i in range(max(0, len(clean) - 1))]
        elif kind == "char3":
            tokens = [clean[i : i + 3] for i in range(max(0, len(clean) - 2))]
        elif kind == "char23":
            tokens = [clean[i : i + n] for n in (2, 3) for i in range(max(0, len(clean) - n + 1))]
        elif kind == "word_char2":
            tokens = [f"w:{x}" for x in tokenize_words(text)] + [
                f"c:{clean[i:i+2]}" for i in range(max(0, len(clean) - 1))
            ]
        elif kind == "pos_char2":
            tokens = [f"p:{x}" for x in baseline_module.ctras_tokens(text, stopwords)] + [
                f"c:{clean[i:i+2]}" for i in range(max(0, len(clean) - 1))
            ]
        else:
            raise ValueError(kind)
        result.append(set(tokens))
    return result


def jaccard_similarity(sets: list[set[str]]) -> np.ndarray:
    sim = np.eye(len(sets), dtype=np.float32)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            value = len(sets[i] & sets[j]) / len(union) if union else 1.0
            sim[i, j] = sim[j, i] = value
    return sim


def tfidf_similarities(records: list[dict]) -> dict[str, np.ndarray]:
    texts = [row["description"] for row in records]
    configs = {
        "tfidf_char23": TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True),
        "tfidf_char24": TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True),
        "tfidf_char35": TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True),
        "tfidf_word12": TfidfVectorizer(
            analyzer=tokenize_words, ngram_range=(1, 2), sublinear_tf=True
        ),
    }
    result = {}
    for name, vectorizer in configs.items():
        matrix = vectorizer.fit_transform(texts)
        result[name] = cosine_similarity(matrix).astype(np.float32)
    return result


def iter_tree_partitions(tree, ids: list[str]):
    
    
    for cut_i in range(1, 100):
        labels = fcluster(tree, t=cut_i / 100, criterion="distance")
        yield f"distance={cut_i/100:.2f}", stable_partition(labels, ids)
    max_k = min(len(ids) - 1, 400)
    for k in range(10, max_k + 1, 5):
        labels = fcluster(tree, t=k, criterion="maxclust")
        yield f"maxclust={k}", stable_partition(labels, ids)


def evaluate_family(method: str, dataset: str, ids: list[str], gold: dict[str, str],
                    matrices: dict[str, np.ndarray], linkages=("average", "complete", "single")):
    rows = []
    best = None
    best_partition = None
    for representation, similarity in matrices.items():
        for linkage_name in linkages:
            tree = tree_from_similarity(similarity, linkage_name)
            for cut, partition in iter_tree_partitions(tree, ids):
                metrics = score(partition, gold)
                row = {
                    "method": method,
                    "dataset": dataset,
                    "representation": representation,
                    "linkage": linkage_name,
                    "cut": cut,
                    **metrics,
                }
                rows.append(row)
                if best is None or (row["ari"], row["b3_f1"]) > (best["ari"], best["b3_f1"]):
                    best = row
                    best_partition = partition
    assert best is not None and best_partition is not None
    return rows, best, best_partition


def load_stopwords(path: Path) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def useful_token(token: str, stopwords: set[str]) -> bool:
    token = token.strip().lower()
    return bool(token and token not in stopwords and re.search(r"[\u3400-\u9fffA-Za-z0-9]", token))


def ctras_tokens(text: str, stopwords: set[str]) -> list[str]:
    """CTRAS-TXT: retain noun/verb tokens after Chinese segmentation/POS."""
    text = normalize_text(text)
    tokens = [
        item.word.strip().lower()
        for item in pseg.cut(text)
        if item.flag and item.flag[0] in {"n", "v"} and useful_token(item.word, stopwords)
    ]
    
    
    
    if not tokens:
        tokens = [w.strip().lower() for w in jieba.lcut(text) if useful_token(w, stopwords)]
    return tokens


def normalized_chars(text: str) -> str:
    return re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", text.lower())


def feature_matrices(records, module, stopwords):
    raw = [row["description"] for row in records]
    clean = [normalized_chars(text) for text in raw]
    words = [" ".join(tuning.tokenize_words(text)) for text in raw]
    pos = [" ".join(module.ctras_tokens(text, stopwords)) for text in raw]
    matrices = {}

    vectorizers = {
        "tfidf_char12_clean": (clean, TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True)),
        "tfidf_char23_clean": (clean, TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True)),
        "tfidf_char24_clean": (clean, TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)),
        "tfidf_char35_clean": (clean, TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)),
        "tfidf_char23_binary": (clean, TfidfVectorizer(analyzer="char", ngram_range=(2, 3), binary=True, use_idf=True)),
        "tfidf_word12": (words, TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", ngram_range=(1, 2), sublinear_tf=True)),
        "tfidf_word13": (words, TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", ngram_range=(1, 3), sublinear_tf=True)),
        "tfidf_pos12": (pos, TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", ngram_range=(1, 2), sublinear_tf=True)),
    }
    for name, (texts, vectorizer) in vectorizers.items():
        matrices[name] = cosine_similarity(vectorizer.fit_transform(texts)).astype(np.float32)

    jaccards = {}
    for kind in ("word", "char2", "char3", "char23", "word_char2", "pos_char2"):
        jaccards[f"jaccard_{kind}"] = tuning.jaccard_similarity(
            tuning.token_sets(records, kind, module, stopwords)
        )
    matrices.update(jaccards)

    
    
    for char_name in ("tfidf_char23_clean", "tfidf_char24_clean", "tfidf_char35_clean"):
        for word_name in ("tfidf_word12", "tfidf_pos12"):
            for char_weight in (0.25, 0.50, 0.75):
                name = f"blend({char_name},{word_name},char={char_weight:.2f})"
                matrices[name] = char_weight * matrices[char_name] + (1.0 - char_weight) * matrices[word_name]
    for base_name in ("tfidf_char23_clean", "tfidf_char24_clean", "tfidf_word12"):
        for jac_name in ("jaccard_char2", "jaccard_word", "jaccard_word_char2"):
            for jac_weight in (0.20, 0.40):
                name = f"blend({base_name},{jac_name},jaccard={jac_weight:.2f})"
                matrices[name] = (1.0 - jac_weight) * matrices[base_name] + jac_weight * matrices[jac_name]
    return matrices


def reconstruct_partition(row, similarity, ids):
    tree = tuning.tree_from_similarity(similarity, row["linkage"])
    criterion, value = row["cut"].split("=", 1)
    labels = tuning.fcluster(
        tree,
        t=float(value) if criterion == "distance" else int(value),
        criterion=criterion,
    )
    return tuning.stable_partition(labels, ids)


def fuzzy_harden(partition, similarity, ids, secondary_ratio):
    index = {rid: i for i, rid in enumerate(ids)}
    cluster_indices = [[index[str(rid)] for rid in group] for group in partition]
    scores = np.zeros((len(ids), len(partition)), dtype=np.float32)
    for cluster_i, members in enumerate(cluster_indices):
        scores[:, cluster_i] = similarity[:, members].mean(axis=1)
    original = np.empty(len(ids), dtype=int)
    for cluster_i, members in enumerate(cluster_indices):
        original[members] = cluster_i
    
    
    
    ranked = np.argsort(-scores, axis=1)
    memberships = []
    hard_labels = []
    native = []
    for i, rid in enumerate(ids):
        best_score = float(scores[i, ranked[i, 0]])
        choices = [int(c) for c in ranked[i, :3] if float(scores[i, c]) >= best_score * secondary_ratio]
        if int(original[i]) not in choices:
            choices.append(int(original[i]))
        choices = sorted(set(choices), key=lambda c: (-float(scores[i, c]), c))[:3]
        chosen = max(choices, key=lambda c: (float(scores[i, c]), c == int(original[i]), -c))
        hard_labels.append(chosen)
        memberships.append(choices)
        native.append(
            {
                "source_id": str(rid),
                "primary_cluster": int(chosen),
                "seed_cluster": int(original[i]),
                "memberships": choices,
                "scores": [float(scores[i, c]) for c in choices],
            }
        )
    return tuning.stable_partition(hard_labels, ids), native

