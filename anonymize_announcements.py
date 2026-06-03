#!/usr/bin/env python3
"""Anonymize Canvas announcement text with per-announcement deterministic aliases.

The algorithm uses one random project seed plus the announcement id, entity type,
and normalized original value. The MD5 digest is converted into a placeholder
that keeps the original entity's rough shape, while repeated mentions inside the
same announcement map to the same replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import string
from dataclasses import dataclass, field


DEFAULT_SEED = "canvas-demo-2026-06-02-local"
LETTERS = string.ascii_uppercase
NAME_SYLLABLES = (
    "lan",
    "ver",
    "min",
    "cor",
    "tav",
    "ren",
    "syl",
    "mar",
    "den",
    "ali",
    "vor",
    "nex",
    "cal",
    "rin",
    "bel",
    "tor",
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
COURSE_RE = re.compile(r"\b[A-Z]{2,4}\d{3,4}[A-Z]?\b")
ZOOM_RE = re.compile(r"https?://(?:[\w.-]+\.)?zoom\.us/[^\s)>\"]+", re.I)
ROOM_RE = re.compile(r"\b[A-Z]{2}\d{3,4}\b")
NAME_HINT_RE = re.compile(
    r"\b(?:Prof\.|Professor|Tutor:|TA:|Warm wishes -|Best,)\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)


def _digest(seed: str, announcement_id: str, kind: str, value: str) -> str:
    normalized = " ".join(value.lower().split())
    raw = f"{seed}|{announcement_id}|{kind}|{normalized}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _letters(hex_digest: str, length: int) -> str:
    number = int(hex_digest, 16)
    chars = []
    for _ in range(length):
        chars.append(LETTERS[number % len(LETTERS)])
        number //= len(LETTERS)
    return "".join(chars)


def _alias_name(hex_digest: str, words: int) -> str:
    number = int(hex_digest, 16)
    aliases = []
    for _ in range(words):
        parts = []
        for _ in range(2 + number % 2):
            parts.append(NAME_SYLLABLES[number % len(NAME_SYLLABLES)])
            number //= len(NAME_SYLLABLES)
        aliases.append("".join(parts).capitalize()[:10])
    return " ".join(aliases)


@dataclass
class AnnouncementAnonymizer:
    seed: str = DEFAULT_SEED
    announcement_id: str = "announcement"
    replacements: dict[tuple[str, str], str] = field(default_factory=dict)

    def replacement(self, kind: str, value: str) -> str:
        key = (kind, " ".join(value.lower().split()))
        if key in self.replacements:
            return self.replacements[key]

        digest = _digest(self.seed, self.announcement_id, kind, value)
        if kind == "email":
            alias = f"user{int(digest[:8], 16) % 90000 + 10000}@{_letters(digest[8:], 3).lower()}.{_letters(digest[14:], 2).lower()}"
        elif kind == "course":
            alias = f"{_letters(digest, 3)}{int(digest[:6], 16) % 9000 + 1000}"
        elif kind == "zoom":
            alias = f"https://zoom.us/j/{int(digest[:12], 16) % 90000000000 + 10000000000}"
        elif kind == "room":
            alias = f"{_letters(digest, 2)}{int(digest[:5], 16) % 900 + 100}"
        elif kind == "name":
            alias = _alias_name(digest, len(value.split()))
        else:
            alias = f"TOKEN_{digest[:8]}"

        self.replacements[key] = alias
        return alias

    def anonymize(self, text: str) -> str:
        text = EMAIL_RE.sub(lambda m: self.replacement("email", m.group(0)), text)
        text = ZOOM_RE.sub(lambda m: self.replacement("zoom", m.group(0)), text)
        text = COURSE_RE.sub(lambda m: self.replacement("course", m.group(0)), text)
        text = ROOM_RE.sub(lambda m: self.replacement("room", m.group(0)), text)

        for match in list(NAME_HINT_RE.finditer(text)):
            original = match.group(1)
            text = re.sub(
                rf"\b{re.escape(original)}\b",
                self.replacement("name", original),
                text,
                flags=re.I,
            )
        return text


def split_announcements(text: str) -> list[tuple[str, str]]:
    pieces = re.split(r"(?m)(?=^#\d+\b)", text)
    return [(piece.splitlines()[0].strip("# "), piece) for piece in pieces if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        text = handle.read()

    output = []
    for announcement_id, piece in split_announcements(text):
        anonymizer = AnnouncementAnonymizer(seed=args.seed, announcement_id=announcement_id)
        output.append(anonymizer.anonymize(piece))
    print("\n".join(output), end="")


if __name__ == "__main__":
    main()
