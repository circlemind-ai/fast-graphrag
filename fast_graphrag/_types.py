import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from ._models import BaseModelAlias, dump_to_csv, dump_to_reference_list

####################################################################################################
# GENERICS
####################################################################################################


@dataclass
class TSerializable:
    F_TO_CONTEXT: ClassVar[List[str]] = []

    @classmethod
    def to_dict(
        cls,
        obj: Optional["TSerializable"] = None,
        objs: Optional[Iterable["TSerializable"]] = None,
        include_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Compute the fields to include
        if include_fields is None:
            include_fields = [f.name for f in fields(cls)]
        if obj is not None:
            assert objs is None, "Either edge or edges should be provided, not both"
            return {f: getattr(obj, f) for f in include_fields}
        elif objs is not None:
            return {f: [getattr(o, f) for o in objs] for f in include_fields}
        return {}


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
class BTNode(TSerializable):
    name: Any


GTNode = TypeVar("GTNode", bound=BTNode)


@dataclass
class BTEdge(TSerializable):
    source: Any
    target: Any

    @staticmethod
    def to_attrs(edge: Optional[Any] = None, edges: Optional[Iterable[Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


GTEdge = TypeVar("GTEdge", bound=BTEdge)


@dataclass
class BTChunk(TSerializable):
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
    F_TO_CONTEXT = ["content", "metadata"]

    id: THash = field()
    content: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


# Graph types
@dataclass
class TEntity(BaseModelAlias, BTNode):
    F_TO_CONTEXT = ["name", "description"]

    name: str = field()
    type: str = field()
    description: str = field()

    def to_str(self) -> str:
        s = f"[{self.type}] {self.name}"
        try:
            if self.description and len(self.description):
                s += f"\n[DESCRIPTION] {self.description}"
        except Exception:
            print(f"{self} cannot call to_str")
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
    F_TO_CONTEXT = ["source", "target", "description"]

    source: str = field()
    target: str = field()
    description: str = field()
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
        # alternative description "Explanation of why the source entity and the target entity are related to each other"
        desc: str = Field(
            ...,
            description="Description of the relationship between the source and target entity",
        )

        @staticmethod
        def to_dataclass(pydantic: "TRelation.Model") -> "TRelation":
            return TRelation(
                source=pydantic.source,
                target=pydantic.target,
                description=pydantic.desc,
            )

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

    entities: List[Tuple[GTNode, TScore]] = field()
    relations: List[Tuple[GTEdge, TScore]] = field()
    chunks: List[Tuple[GTChunk, TScore]] = field()

    def truncate(self, max_chars: Dict[str, int], output_context_str: bool = False) -> str:
        """Genearate a tabular representation of the context.

        Truncate the tables to the maximum number of assigned tokens.
        """
        csv_tables: Dict[str, List[str]] = {
            "entities": dump_to_csv([e for e, _ in self.entities], ["name", "description"], with_header=True),
            "relations": dump_to_csv(
                [r for r, _ in self.relations],
                ["source", "target", "description"],
                with_header=True,
            ),
            "chunks": dump_to_reference_list([str(c) for c, _ in self.chunks]),
        }
        csv_tables_row_length = {k: [len(row) for row in table] for k, table in csv_tables.items()}

        # Truncate each csv to the maximum number of assigned tokens
        included_up_to = {key: 0 for key in ["entities", "relations", "chunks"]}
        chars_remainder = 0
        while True:
            last_char_remainder = chars_remainder
            # Keep augmenting the context until feasible
            for table in csv_tables:
                for i in range(included_up_to[table], len(csv_tables_row_length[table])):
                    length = csv_tables_row_length[table][i] + 1  # +1 for the newline character
                    if (
                        (length <= chars_remainder) and (max_chars[table] >= 0)
                    ):  # use up the remainder (handle -1 differently)
                        included_up_to[table] += 1
                        chars_remainder -= length
                        break
                    elif length <= max_chars[table]:  # use up the assigned tokens
                        included_up_to[table] += 1
                        max_chars[table] -= length
                    else:
                        break

                if max_chars[table] >= 0:  # if the assigned tokens are not used up store in the remainder
                    chars_remainder += max_chars[table]
                    max_chars[table] = 0

            # Truncate the csv
            if chars_remainder == last_char_remainder:
                break

        # Truncate the context
        self.entities = self.entities[: included_up_to["entities"]]
        self.relations = self.relations[: included_up_to["relations"]]
        self.chunks = self.chunks[: included_up_to["chunks"]]

        # Generate the context string
        context: List[str] = []
        if output_context_str:
            if len(self.entities):
                context.extend(
                    [
                        "\n## Entities",
                        "```csv",
                        *csv_tables["entities"][: included_up_to["entities"]],
                        "```",
                    ]
                )
            else:
                context.append("\n## Entities: Not provided\n")

            if len(self.relations):
                context.extend(
                    [
                        "\n## Relationships",
                        "```csv",
                        *csv_tables["relations"][: included_up_to["relations"]],
                        "```",
                    ]
                )
            else:
                context.append("\n## Relationships: Not provided\n")

            if len(self.chunks):
                context.extend(
                    [
                        "\n## Sources\n",
                        *csv_tables["chunks"][: included_up_to["chunks"]],
                        "",
                    ]
                )
            else:
                context.append("\n## Sources: Not provided\n")
        return "\n".join(context)


@dataclass
class TQueryResponse(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing a query response."""

    response: str
    context: TContext[GTNode, GTEdge, GTHash, GTChunk]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the query response to a dictionary."""
        return {
            "response": self.response,
            "context": {
                "entities": [(e.to_dict(e, include_fields=e.F_TO_CONTEXT), float(s)) for e, s in self.context.entities],
                "relations": [
                    (r.to_dict(r, include_fields=r.F_TO_CONTEXT), float(s)) for r, s in self.context.relations
                ],
                "chunks": [(c.to_dict(c, include_fields=c.F_TO_CONTEXT), float(s)) for c, s in self.context.chunks],
            },
        }

    # All the machinery to format references
    ####################################################################################################

    @dataclass
    class _Chunk:
        id: int = field()
        content: str = field()
        index: Optional[int] = field(init=False, default=None)

    @dataclass
    class _Document:
        metadata: Dict[str, Any] = field(init=False, default_factory=dict)
        chunks: Dict[int, "TQueryResponse._Chunk"] = field(init=False, default_factory=dict)
        index: Optional[int] = field(init=False, default=None)
        _last_chunk_index: int = field(init=False, default=0)

        def get_chunk(self, id: int) -> Tuple[int, "TQueryResponse._Chunk"]:
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
    class _ReferenceList:
        documents: Dict[int, "TQueryResponse._Document"] = field(
            default_factory=lambda: defaultdict(lambda: TQueryResponse._Document())
        )
        _last_document_index: int = field(init=False, default=0)

        def get_doc(self, id: int) -> Tuple[int, "TQueryResponse._Document"]:
            doc = self.documents[id]
            if doc.index is None:
                self._last_document_index += 1
                doc.index = self._last_document_index
            return doc.index, doc

        def to_dict(self):
            return {doc.index: doc.to_dict() for doc in self.documents.values() if doc.index is not None}

    def format_references(
        self,
        format_fn: Callable[[int, List[int], Any], str] = lambda i, _, __: f"[{i}]",
    ):
        # Create list of documents
        reference_list = self._ReferenceList()
        ref2data: Dict[str, Tuple[int, int]] = {}

        for i, (chunk, _) in enumerate(self.context.chunks):
            metadata: Dict[str, Any] = getattr(chunk, "metadata", {})
            chunk_id = int(chunk.id)
            if metadata == {}:
                doc_id = chunk_id
            else:
                doc_id = hash(frozenset(metadata.items()))
            reference_list.documents[doc_id].metadata = metadata
            reference_list.documents[doc_id].chunks[chunk_id] = TQueryResponse._Chunk(chunk_id, str(chunk))
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

                doc_index, doc = reference_list.get_doc(doc_id)
                r += format_fn(doc_index, [doc.get_chunk(id)[0] for id in chunk_ids], doc.metadata)
            return r

        return (
            re.sub(r"\[\d[\s\d\]\[]*\]", _replace_fn, self.response),
            reference_list.to_dict(),
        )

    ####################################################################################################
