import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


try:
    from numpy.lib.stride_tricks import sliding_window_view as _slidingWindowView
except Exception:
    _slidingWindowView = None


class Tokenizer:
    def __init__(self, pattern: str | None = None):
        if pattern is None:
            pattern = r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\s\w]"
        self._re = re.compile(pattern, re.IGNORECASE)

    def tokenize(self, text: str) -> list[str]:
        return self._re.findall(text.lower())


def loadNewsTexts(path: str | Path, maxRows: int = 7000) -> list[str]:
    out: list[str] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= int(maxRows):
                break
            obj = json.loads(line)
            headline = (obj.get("headline") or "").strip()
            desc = (obj.get("short_description") or "").strip()
            s = (headline + " " + desc).strip()
            if s:
                out.append(s)
    return out


def loadAmazonTexts(path: str | Path, maxRows: int = 7000) -> list[str]:
    out: list[str] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i >= int(maxRows):
                break
            if not row:
                continue
            title = row[1].strip() if len(row) > 1 else ""
            body = row[2].strip() if len(row) > 2 else ""
            s = (title + " " + body).strip()
            if s:
                out.append(s)
    return out


def textsToTokenStream(
    texts: list[str],
    tokenizer: Tokenizer | None = None,
    maxTokens: int | None = 90000,
    bosToken: str = "<bos>",
    eosToken: str = "<eos>",
) -> list[str]:
    if tokenizer is None:
        tokenizer = Tokenizer()
    stream: list[str] = []
    for s in texts:
        stream.append(bosToken)
        stream.extend(tokenizer.tokenize(s))
        stream.append(eosToken)
        if maxTokens is not None and len(stream) >= int(maxTokens):
            stream = stream[: int(maxTokens)]
            break
    return stream


@dataclass(frozen=True)
class Vocabulary:
    wordToIdx: dict[str, int]
    idxToWord: dict[int, str]
    padToken: str = "<pad>"
    unkToken: str = "<unk>"
    bosToken: str = "<bos>"
    eosToken: str = "<eos>"

    @classmethod
    def fromTokens(
        cls,
        tokens: list[str],
        maxVocab: int = 1200,
        minFreq: int = 2,
        padToken: str = "<pad>",
        unkToken: str = "<unk>",
        bosToken: str = "<bos>",
        eosToken: str = "<eos>",
    ) -> "Vocabulary":
        specials = [padToken, unkToken, bosToken, eosToken]
        counts = Counter(tokens)
        vocab = specials[:]
        for w, c in counts.most_common():
            if w in specials:
                continue
            if int(c) < int(minFreq):
                continue
            vocab.append(w)
            if len(vocab) >= int(maxVocab):
                break
        wordToIdx = {w: i for i, w in enumerate(vocab)}
        idxToWord = {i: w for w, i in wordToIdx.items()}
        return cls(
            wordToIdx=wordToIdx,
            idxToWord=idxToWord,
            padToken=padToken,
            unkToken=unkToken,
            bosToken=bosToken,
            eosToken=eosToken,
        )

    def __len__(self) -> int:
        return len(self.wordToIdx)

    def encode(self, tokens: list[str]) -> np.ndarray:
        unkId = self.wordToIdx[self.unkToken]
        return np.array([self.wordToIdx.get(w, unkId) for w in tokens], dtype=np.int64)

    def decode(self, ids: list[int] | np.ndarray) -> list[str]:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return [self.idxToWord.get(int(i), self.unkToken) for i in ids]

    def decodeText(self, ids: list[int] | np.ndarray, sep: str = " ") -> str:
        return sep.join(self.decode(ids))


def trainValSplit(tokenIds: np.ndarray, valRatio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    tokenIds = np.asarray(tokenIds, dtype=np.int64)
    split = int((1.0 - float(valRatio)) * len(tokenIds))
    return tokenIds[:split], tokenIds[split:]


def makeNgramXY(tokenIds: np.ndarray, contextSize: int) -> tuple[np.ndarray, np.ndarray]:
    tokenIds = np.asarray(tokenIds, dtype=np.int64)
    c = int(contextSize)
    if c <= 0:
        raise ValueError("contextSize must be >= 1")
    if len(tokenIds) <= c:
        return np.zeros((0, c), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    if _slidingWindowView is None:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for i in range(c, len(tokenIds)):
            xs.append(tokenIds[i - c : i])
            ys.append(int(tokenIds[i]))
        return np.stack(xs).astype(np.int64, copy=False), np.array(ys, dtype=np.int64)
    w = _slidingWindowView(tokenIds, c + 1)
    return w[:, :c].astype(np.int64, copy=False), w[:, -1].astype(np.int64, copy=False)


def makeRnnXY(tokenIds: np.ndarray, seqLen: int, stride: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    tokenIds = np.asarray(tokenIds, dtype=np.int64)
    t = int(seqLen)
    if t <= 0:
        raise ValueError("seqLen must be >= 1")
    if stride is None:
        stride = t
    s = int(stride)
    if len(tokenIds) <= t + 1:
        return np.zeros((0, t), dtype=np.int64), np.zeros((0, t), dtype=np.int64)
    if _slidingWindowView is None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for i in range(0, len(tokenIds) - t - 1, s):
            xs.append(tokenIds[i : i + t])
            ys.append(tokenIds[i + 1 : i + t + 1])
        return np.stack(xs).astype(np.int64, copy=False), np.stack(ys).astype(np.int64, copy=False)
    w = _slidingWindowView(tokenIds, t + 1)[::s]
    return w[:, :t].astype(np.int64, copy=False), w[:, 1:].astype(np.int64, copy=False)


def sampleXY(x: np.ndarray, y: np.ndarray, maxSamples: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if maxSamples is None or len(x) <= int(maxSamples):
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(x), size=int(maxSamples), replace=False)
    return x[idx], y[idx]
