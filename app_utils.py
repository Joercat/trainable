
import re, json, math, collections, itertools, os, hashlib, time, random

class SimpleTokenizer:
    def __init__(self, bpe_merges=10000):
        try:
            from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
            self.hf = True
        except Exception:
            self.hf = False
        self.vocab = {}
        self.inv_vocab = {}
        self.bpe_merges = bpe_merges

    def normalize(self, text):
        return text.replace("\r","").strip()

    def tokenize(self, text):
        text = self.normalize(text)
        if self.hf:
            return text.split()
        toks = re.findall(r"\w+|[^\s\w]", text)
        return toks

    def build_bpe(self, corpus_lines):
        freqs = collections.Counter()
        for ln in corpus_lines:
            toks = list(self.tokenize(ln))
            for i in range(len(toks)-1):
                freqs[(toks[i], toks[i+1])] += 1
        merges = [p for p,_ in freqs.most_common(self.bpe_merges)]
        self.bpe_merges_list = merges
        return merges

class InterpolatedKneserNey:
    """
    A lightweight, CPU-friendly approximated Interpolated Kneser-Ney style n-gram model.
    This is intentionally simplified to run on Render free tier but produces stronger
    predictive behavior than plain counts via discounting and interpolation.
    """
    def __init__(self, n=3, discount=0.75):
        self.n = n
        self.discount = discount
        # counts[k] maps ngram tuple -> count, where k is ngram length
        self.counts = [collections.Counter() for _ in range(self.n+1)]
        # continuations: for lower-order Kneser-Ney-style probability
        self.continuation_counts = [collections.Counter() for _ in range(self.n+1)]
        self.context_counts = [collections.Counter() for _ in range(self.n+1)]
        self.total_unigrams = 0
        self.vocab = set()

    def _add_ngram(self, ng):
        k = len(ng)
        self.counts[k][tuple(ng)] += 1
        ctx = tuple(ng[:-1]) if k>1 else ()
        self.context_counts[k][ctx] += 1
        # for continuation counts: record that the last token follows the history token
        if k>1:
            self.continuation_counts[k-1][ng[-1]] += 1

    def train_lines(self, lines):
        for ln in lines:
            toks = ln.split()
            toks = ["<s>"] + toks + ["</s>"]
            for k in range(1, self.n+1):
                for i in range(len(toks)-k+1):
                    ng = toks[i:i+k]
                    self._add_ngram(ng)
                    for t in ng:
                        self.vocab.add(t)
        self.total_unigrams = sum(self.counts[1].values())

    def save(self, path):
        data = {"n": self.n, "discount": self.discount, "counts": [], "context_counts": [], "vocab": list(self.vocab)}
        for k in range(len(self.counts)):
            # convert Counter with tuple keys into serializable form
            data["counts"].append({ " ".join(k): v for k,v in self.counts[k].items() })
            data["context_counts"].append({ " ".join(k): v for k,v in self.context_counts[k].items() })
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.n = data.get("n",3)
        self.discount = data.get("discount",0.75)
        self.counts = [collections.Counter({tuple(k.split(" ")):v for k,v in d.items()}) for d in data["counts"]]
        self.context_counts = [collections.Counter({tuple(k.split(" ")):v for k,v in d.items()}) for d in data["context_counts"]]
        self.vocab = set(data.get("vocab", []))
        self.total_unigrams = sum(self.counts[1].values())

    def _p_continuation(self, token):
        # lower-order continuation probability (approx)
        denom = max(1, sum(self.continuation_counts[1].values()))
        return self.continuation_counts[1].get(token, 0) / denom

    def _prob_kn(self, ng):
        """
        Compute a smoothed probability for an ngram tuple using a simplified
        interpolated Kneser-Ney formula.
        """
        k = len(ng)
        if k==1:
            count = self.counts[1].get(tuple(ng), 0)
            return (count / max(1, self.total_unigrams))
        ctx = tuple(ng[:-1])
        count_ng = self.counts[k].get(tuple(ng), 0)
        count_ctx = self.context_counts[k].get(ctx, 0)
        D = self.discount
        first_term = max(count_ng - D, 0) / max(1, count_ctx)
        # lambda for interpolation
        unique_continuations = sum(1 for ng2 in self.counts[k] if ng2[:-1]==ctx)
        lamb = (D * unique_continuations) / max(1, count_ctx)
        lower = self._prob_kn(ng[1:]) if k>1 else self._p_continuation(ng[-1])
        return first_term + lamb * lower

    def predict(self, text, max_len=40):
        toks = text.split()
        if not toks:
            return ""
        out = []
        for _ in range(max_len):
            # build context up to n-1 tokens
            ctx = toks[-(self.n-1):] if len(toks)>=1 else []
            candidates = {}
            # gather candidate next tokens from counts of order n down to 1
            for k in range(self.n,0,-1):
                ctx_k = tuple(ctx[-(k-1):]) if k>1 else ()
                for ng,cnt in self.counts[k].items():
                    if k==1 or ng[:-1]==ctx_k:
                        candidates[ng[-1]] = candidates.get(ng[-1], 0) + cnt
            if not candidates:
                break
            # score candidates with KN-like probability where possible
            scored = []
            for tok,cnt in candidates.items():
                ng = tuple((ctx + [tok])[-(self.n):])
                p = self._prob_kn(list(ng))
                scored.append((p, tok))
            scored.sort(reverse=True)
            token = scored[0][1]
            if token == "</s>": break
            out.append(token)
            toks.append(token)
        return " ".join(out)

# Utilities
def sanitize_text(s):
    return s.replace("\n"," ").strip()

def dedupe_lines(lines):
    seen = set(); out = []
    for ln in lines:
        h = hashlib.sha256(ln.encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); out.append(ln)
    return out
