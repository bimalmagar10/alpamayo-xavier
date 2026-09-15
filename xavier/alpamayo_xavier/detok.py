"""Turn reasoning token ids back into the text the model actually wrote.

The inference environment does not need transformers: detokenising a byte-level
BPE vocabulary is a table lookup and a UTF-8 decode, and the table is just the list
of surface forms, one per id, which h100/a8_token_strings.py writes to
fixtures/vocab.json.

Byte-level BPE (GPT-2's scheme, which Qwen inherits) stores every token as a string
of *printable* characters standing in for raw bytes, so that a vocabulary entry can
never contain a control character or a bare space. Decoding reverses that mapping
and only then interprets the bytes as UTF-8 -- which is why ids must be decoded as
a group: one multi-byte character is often split across two tokens, and decoding
those tokens separately would produce two replacement characters instead of one é.
"""
from __future__ import print_function

import json
import hashlib
import os


def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode: printable stand-in character -> byte."""
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1)))
    codes, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            codes.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in codes], keep))


class Detokenizer(object):
    """vocab.json -> text. `available` is False when the fixture was not shipped."""

    def __init__(self, vocab=None, source=None, error=None):
        self.vocab = vocab or []
        self.source = source
        self.back = _byte_decoder()
        self.error = error

    @property
    def available(self):
        return bool(self.vocab)

    @classmethod
    def find(cls, *dirs):
        """The first fixtures/vocab.json among `dirs`; an empty Detokenizer if none."""
        errors = []
        for d in dirs:
            if not d:
                continue
            path = d if d.endswith(".json") else os.path.join(d, "vocab.json")
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        doc = json.load(f)
                    if isinstance(doc, dict) and doc.get("encoding") != "byte-level-bpe/gpt2":
                        raise ValueError("unsupported vocabulary encoding")
                    tokens = doc.get("tokens") if isinstance(doc, dict) else doc
                    if not isinstance(tokens, list) or not all(t is None or isinstance(t, str) for t in tokens):
                        raise ValueError("tokens must be a list of strings or null")
                except (ValueError, OSError) as exc:
                    errors.append("%s: %s" % (path, exc))
                    continue
                return cls(tokens, path)
        return cls(error="; ".join(errors) or "vocab.json not found")

    def metadata(self):
        result = dict(available=self.available, source=self.source, error=self.error,
                      encoding="byte-level-bpe/gpt2", entries=len(self.vocab))
        if self.source:
            with open(self.source, "rb") as f:
                result["sha256"] = hashlib.sha256(f.read()).hexdigest()
        return result

    def _bytes(self, ids):
        out = bytearray()
        for i in ids:
            piece = self.vocab[i] if 0 <= i < len(self.vocab) else None
            if piece is None:                       # a hole in a padded vocabulary
                out += ("<unused-%d>" % i).encode("utf-8")
                continue
            for ch in piece:
                b = self.back.get(ch)
                # A character outside the stand-in table means the exporter wrote a
                # literal (an added token). Take it at face value, as its own bytes.
                out += bytes([b]) if b is not None else ch.encode("utf-8")
        return bytes(out)

    def decode(self, ids):
        """The whole sequence as one string -- the only correct way to read it."""
        return self._bytes(ids).decode("utf-8", "replace") if self.available else ""

    def pieces(self, ids):
        """Per-token strings, so token boundaries stay visible in the results file.

        A token that holds half of a multi-byte character decodes to U+FFFD here and
        reads correctly only in `decode()`. Both are recorded for exactly that reason.
        """
        return [self._bytes([i]).decode("utf-8", "replace") for i in ids] if self.available else []
