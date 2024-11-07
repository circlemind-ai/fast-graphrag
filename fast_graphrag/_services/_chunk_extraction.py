import re
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterable, List, Set, Tuple

import xxhash

from fast_graphrag._types import TChunk, TDocument, THash
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO

from ._base import BaseChunkingService

DEFAULT_SEPARATORS = [
    # Paragraph and page separators
    "\n\n\n",
    "\n\n",
    "\r\n\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
]


@dataclass
class DefaultChunkingServiceConfig:
    separators: List[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    chunk_token_size: int = field(default=1024)
    chunk_token_overlap: int = field(default=128)


@dataclass
class DefaultChunkingService(BaseChunkingService[TChunk]):
    """Default class for chunk extractor."""

    config: DefaultChunkingServiceConfig = field(default_factory=DefaultChunkingServiceConfig)

    def __post_init__(self):
        self._split_re = re.compile(f"({'|'.join(re.escape(s) for s in self.config.separators or [])})")
        self._chunk_size = self.config.chunk_token_size * TOKEN_TO_CHAR_RATIO
        self._chunk_overlap = self.config.chunk_token_overlap * TOKEN_TO_CHAR_RATIO

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[TChunk]]:
        """Extract unique chunks from the given data."""
        chunks_per_data: List[List[TChunk]] = []

        for d in data:
            unique_chunk_ids: Set[THash] = set()
            extracted_chunks = await self._extract_chunks(d)
            chunks: List[TChunk] = []
            for chunk in extracted_chunks:
                if chunk.id not in unique_chunk_ids:
                    unique_chunk_ids.add(chunk.id)
                    chunks.append(chunk)
            chunks_per_data.append(chunks)

        return chunks_per_data

    async def _extract_chunks(self, data: TDocument) -> List[TChunk]:
        # Sanitize input
        data.data = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", data.data)

        if len(data.data) <= self._chunk_size:
            chunks = [data.data]
        else:
            chunks = self._split_text(data.data)

        return [
            TChunk(
                id=THash(xxhash.xxh3_64_intdigest(chunk)),
                content=chunk,
                metadata=data.metadata,
            )
            for chunk in chunks
        ]

    def _split_text(self, text: str) -> List[str]:
        return self._merge_splits(self._split_re.split(text))

    def _merge_splits(self, splits: List[str]) -> List[str]:
        if not splits:
            return []

        # Add empty string to the end to have a separator at the end of the last chunk
        splits.append("")

        merged_splits: List[List[Tuple[str, int]]] = []
        current_chunk: List[Tuple[str, int]] = []
        current_chunk_length: int = 0

        for i, split in enumerate(splits):
            split_length: int = len(split)
            # Ignore splitting if it's a separator
            if (i % 2 == 1) or (
                current_chunk_length + split_length <= self._chunk_size - (self._chunk_overlap if i > 0 else 0)
            ):
                current_chunk.append((split, split_length))
                current_chunk_length += split_length
            else:
                merged_splits.append(current_chunk)
                current_chunk = [(split, split_length)]
                current_chunk_length = split_length

        merged_splits.append(current_chunk)

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        else:
            r = ["".join((c[0] for c in chunk)) for chunk in merged_splits]

        return r

    def _enforce_overlap(self, chunks: List[List[Tuple[str, int]]]) -> List[str]:
        result: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append("".join((c[0] for c in chunk)))
            else:
                # Compute overlap
                overlap_length: int = 0
                overlap: List[str] = []
                for text, length in reversed(chunks[i - 1]):
                    if overlap_length + length > self._chunk_overlap:
                        break
                    overlap_length += length
                    overlap.append(text)
                result.append("".join(chain(reversed(overlap), (c[0] for c in chunk))))
        return result
