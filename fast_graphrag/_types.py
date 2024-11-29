import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from ._models import BaseModelAlias, dump_to_csv, dump_to_reference_list

####################################################################################################
# GENERICS
####################################################################################################

# Blob
GTBlob = TypeVar("GTBlob")

# KeyValue
GTKey = TypeVar("GTKey")
GTValue = TypeVar("GTValue")

# Vectordb
GTEmbedding = TypeVar("GTEmbedding")
GTHash = TypeVar("GTHash")

# Graph
GTId = TypeVar("GTId")


@dataclass
class BTNode:
    name: Any


GTNode = TypeVar("GTNode", bound=BTNode)


@dataclass
class BTEdge:
    source: Any
    target: Any

    @staticmethod
    def to_attrs(edge: Optional[Any] = None, edges: Optional[Iterable[Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


GTEdge = TypeVar("GTEdge", bound=BTEdge)


@dataclass
class BTChunk:
    id: Any


GTChunk = TypeVar("GTChunk", bound=BTChunk)


####################################################################################################
# TYPES
####################################################################################################

# Embedding types
TEmbeddingType: TypeAlias = np.float32
TEmbedding: TypeAlias = npt.NDArray[TEmbeddingType]

THash: TypeAlias = np.uint64
TScore: TypeAlias = np.float32
TIndex: TypeAlias = int
TId: TypeAlias = str


@dataclass
class TDocument:
    """A class for representing a piece of data."""

    data: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TChunk(BTChunk):
    """A class for representing a chunk in a TDocument."""

    id: THash = field()
    content: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


# Graph types
@dataclass
class TEntity(BaseModelAlias, BTNode):
    name: str = field()
    type: str = field()
    description: str = field()

    def to_str(self) -> str:
        s = f"[NAME] {self.name}"
        if len(self.description):
            s += f"  [DESCRIPTION] {self.description}"
        return s

    class Model(BaseModelAlias.Model, alias="Entity"):
        name: str = Field(..., description="Name of the entity")
        type: str = Field(..., description="Type of the entity")
        desc: str = Field(..., description="Description of the entity")

        @staticmethod
        def to_dataclass(pydantic: "TEntity.Model") -> "TEntity":
            return TEntity(name=pydantic.name, type=pydantic.type, description=pydantic.desc)

        @field_validator("name", mode="before")
        @classmethod
        def uppercase_name(cls, value: str):
            return value.upper() if value else value

        @field_validator("type", mode="before")
        @classmethod
        def uppercase_type(cls, value: str):
            return value.upper() if value else value


@dataclass
class TRelation(BaseModelAlias, BTEdge):
    source: str = field()
    target: str = field()
    description: str = field()
    keywords: List[str] = field()
    chunks: List[THash] | None = field(default=None)

    @staticmethod
    def to_attrs(
        edge: Optional["TRelation"] = None,
        edges: Optional[Iterable["TRelation"]] = None,
        include_source_target: bool = False,
        **_,
    ) -> Dict[str, Any]:
        if edge is not None:
            assert edges is None, "Either edge or edges should be provided, not both"
            return {
                "description": edge.description,
                "keywords": edge.keywords,
                "chunks": edge.chunks,
                **(
                    {
                        "source": edge.source,
                        "target": edge.target,
                    }
                    if include_source_target
                    else {}
                ),
            }
        elif edges is not None:
            return {
                "description": [e.description for e in edges],
                "keywords": [e.keywords for e in edges],
                "chunks": [e.chunks for e in edges],
                **(
                    {
                        "source": [e.source for e in edges],
                        "target": [e.target for e in edges],
                    }
                    if include_source_target
                    else {}
                ),
            }
        else:
            return {}

    class Model(BaseModelAlias.Model, alias="Relationship"):
        source: str = Field(..., description="Name of the source entity")
        target: str = Field(..., description="Name of the target entity")
        keywords: List[str] = Field(..., description="Keywords associated with the relationship")
        # alternative description "Explanation of why the source entity and the target entity are related to each other"
        desc: str = Field(..., description="Description of the relationship between the source and target entity")

        @staticmethod
        def to_dataclass(pydantic: "TRelation.Model") -> "TRelation":
            return TRelation(source=pydantic.source, target=pydantic.target, keywords=pydantic.keywords, description=pydantic.desc)

        @field_validator("source", mode="before")
        @classmethod
        def uppercase_source(cls, value: str):
            return value.upper() if value else value

        @field_validator("target", mode="before")
        @classmethod
        def uppercase_target(cls, value: str):
            return value.upper() if value else value


@dataclass
class TGraph(BaseModelAlias):
    entities: List[TEntity] = field()
    relationships: List[TRelation] = field()

    class Model(BaseModelAlias.Model, alias="Graph"):
        entities: List[TEntity.Model] = Field(description="List of extracted entities")
        relationships: List[TRelation.Model] = Field(description="Relationships between the entities")
        other_relationships: List[TRelation.Model] = Field(
            description=(
                "Other relationships between the extracted entities previously missed"
                "(likely involving minor/generic entities)"
            )
        )

        @staticmethod
        def to_dataclass(pydantic: "TGraph.Model") -> "TGraph":
            return TGraph(
                entities=[p.to_dataclass(p) for p in pydantic.entities],
                relationships=[p.to_dataclass(p) for p in pydantic.relationships]
                + [p.to_dataclass(p) for p in pydantic.other_relationships],
            )


@dataclass
class TContext(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing the context used to generate a query response."""

    entities: List[Tuple[GTNode, TScore]]
    relationships: List[Tuple[GTEdge, TScore]]
    chunks: List[Tuple[GTChunk, TScore]]

    def to_str(self, max_chars: Dict[str, int]) -> str:
        """Convert the context to a string representation."""
        csv_tables: Dict[str, List[str]] = {
            "entities": dump_to_csv([e for e, _ in self.entities], ["name", "description"], with_header=True),
            "relationships": dump_to_csv(
                [r for r, _ in self.relationships], ["source", "target", "description"], with_header=True
            ),
            "chunks": dump_to_reference_list([str(c) for c, _ in self.chunks]),
        }
        csv_tables_row_length = {k: [len(row) for row in table] for k, table in csv_tables.items()}

        include_up_to = {
            "entities": 0,
            "relationships": 0,
            "chunks": 0,
        }

        # Truncate each csv to the maximum number of assigned tokens
        chars_remainder = 0
        while True:
            last_char_remainder = chars_remainder
            # Keep augmenting the context until feasible
            for table in csv_tables:
                for i in range(include_up_to[table], len(csv_tables_row_length[table])):
                    length = csv_tables_row_length[table][i] + 1  # +1 for the newline character
                    if length <= chars_remainder:  # use up the remainder
                        include_up_to[table] += 1
                        chars_remainder -= length
                    elif length <= max_chars[table]:  # use up the assigned tokens
                        include_up_to[table] += 1
                        max_chars[table] -= length
                    else:
                        break

                if max_chars[table] >= 0:  # if the assigned tokens are not used up store in the remainder
                    chars_remainder += max_chars[table]
                    max_chars[table] = 0

            # Truncate the csv
            if chars_remainder == last_char_remainder:
                break

        data: List[str] = []
        if len(self.entities):
            data.extend(
                [
                    "\n## Entities",
                    "```csv",
                    *csv_tables["entities"][: include_up_to["entities"]],
                    "```",
                ]
            )
        else:
            data.append("\n#Entities: None\n")

        if len(self.relationships):
            data.extend(
                [
                    "\n## Relationships",
                    "```csv",
                    *csv_tables["relationships"][: include_up_to["relationships"]],
                    "```",
                ]
            )
        else:
            data.append("\n## Relationships: None\n")

        if len(self.chunks):
            data.extend(
                [
                    "\n## Sources\n",
                    *csv_tables["chunks"][: include_up_to["chunks"]],
                ]
            )
        else:
            data.append("\n## Sources: None\n")
        return "\n".join(data)


@dataclass
class TQueryResponse(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing a query response."""

    response: str
    context: TContext[GTNode, GTEdge, GTHash, GTChunk]

    @dataclass
    class Chunk:
        id: int = field()
        content: str = field()
        index: Optional[int] = field(init=False, default=None)

    @dataclass
    class Document:
        metadata: Dict[str, Any] = field(init=False, default_factory=dict)
        chunks: Dict[int, "TQueryResponse.Chunk"] = field(init=False, default_factory=dict)
        index: Optional[int] = field(init=False, default=None)
        _last_chunk_index: int = field(init=False, default=0)

        def get_chunk(self, id: int) -> Tuple[int, "TQueryResponse.Chunk"]:
            chunk = self.chunks[id]
            if chunk.index is None:
                self._last_chunk_index += 1
                chunk.index = self._last_chunk_index
            return chunk.index, chunk

        def to_dict(self) -> Dict[str, Any]:
            return {
                "meta": self.metadata,
                "chunks": {
                    chunk.index: (chunk.content, chunk.id) for chunk in self.chunks.values() if chunk.index is not None
                },
            }

    @dataclass
    class Context:
        documents: Dict[int, "TQueryResponse.Document"] = field(
            default_factory=lambda: defaultdict(lambda: TQueryResponse.Document())
        )
        _last_document_index: int = field(init=False, default=0)

        def get_doc(self, id: int) -> Tuple[int, "TQueryResponse.Document"]:
            doc = self.documents[id]
            if doc.index is None:
                self._last_document_index += 1
                doc.index = self._last_document_index
            return doc.index, doc

        def to_dict(self):
            return {doc.index: doc.to_dict() for doc in self.documents.values() if doc.index is not None}

    def format_references(self, format_fn: Callable[[int, List[int], Any], str] = lambda i, _, __: f"[{i}]"):
        # Create list of documents
        context = self.Context()
        ref2data: Dict[str, Tuple[int, int]] = {}

        for i, (chunk, _) in enumerate(self.context.chunks):
            metadata: Dict[str, Any] = getattr(chunk, "metadata", {})
            chunk_id = int(chunk.id)
            if metadata == {}:
                doc_id = chunk_id
            else:
                doc_id = hash(frozenset(metadata.items()))
            context.documents[doc_id].metadata = metadata
            context.documents[doc_id].chunks[chunk_id] = TQueryResponse.Chunk(chunk_id, str(chunk))
            ref2data[str(i + 1)] = (doc_id, chunk_id)

        def _replace_fn(match: str | re.Match[str]) -> str:
            text = match if isinstance(match, str) else match.group()
            references = re.findall(r"(\d+)", text)
            seen_docs: Dict[int, List[int]] = defaultdict(list)

            for reference in references:
                d = ref2data.get(reference, None)
                if d is None:
                    continue
                seen_docs[d[0]].append(d[1])

            r = ""
            for reference in references:
                d = ref2data.get(reference, None)
                if d is None:
                    continue

                doc_id = d[0]
                chunk_ids = seen_docs.get(doc_id, None)
                if chunk_ids is None:
                    continue
                seen_docs.pop(doc_id)

                doc_index, doc = context.get_doc(doc_id)
                r += format_fn(doc_index, [doc.get_chunk(id)[0] for id in chunk_ids], doc.metadata)
            return r

        return re.sub(r"\[\d[\s\d\]\[]*\]", _replace_fn, self.response), context.to_dict()
