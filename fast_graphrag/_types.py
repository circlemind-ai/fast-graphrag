from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator
from pydantic._internal import _model_construction

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
GTGraph = TypeVar("GTGraph")
GTId = TypeVar("GTId")


@dataclass
class BTNode:
    name: Any
    description: Any


GTNode = TypeVar("GTNode", bound=BTNode)


@dataclass
class BTEdge:
    source: Any
    target: Any
    description: Any


GTEdge = TypeVar("GTEdge", bound=BTEdge)


@dataclass
class BTChunk:
    content: Any


GTChunk = TypeVar("GTChunk", bound=BTChunk)


# LLM
def _schema_no_title(schema: dict[str, Any]) -> None:
    schema.pop("required")
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


class MetaModel(_model_construction.ModelMetaclass):
    def __new__(
        cls, name: str, bases: tuple[type[Any], ...], dct: Dict[str, Any], alias: Optional[str] = None, **kwargs: Any
    ) -> type:
        if alias:
            dct["__qualname__"] = alias
        if "BaseModel" not in [base.__name__ for base in bases]:
            bases = bases + (BaseModel,)
        return super().__new__(cls, name, bases, dct, json_schema_extra=_schema_no_title, **kwargs)


class BTResponseModel:
    class Model(BaseModel):
        @staticmethod
        def to_dataclass(pydantic: Any) -> Any:
            raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError


GTResponseModel = TypeVar("GTResponseModel", bound=Union[str, BaseModel, BTResponseModel])

####################################################################################################
# TYPES
####################################################################################################


def dump_to_csv(
    data: Iterable[object],
    fields: List[str],
    separator: str = ";\t",
    with_header: bool = False,
    with_index: bool = False,
    **values: List[Any],
) -> str:
    index_field = ["id"] if with_index else []
    rows = chain(
        (separator.join(chain(index_field, fields, values.keys())),) if with_header else (),
        chain(
            separator.join(
                chain(
                    (str(i + 1),),
                    (str(getattr(d, field)).replace("\n", "  ") for field in fields),
                    (str(v) for v in vs),
                )
            )
            for i, (d, *vs) in enumerate(zip(data, *values.values()))
        )
        if with_index
        else chain(
            separator.join(
                chain(
                    (str(getattr(d, field)).replace("\n", "  ") for field in fields),
                    (str(v) for v in vs),
                )
            )
            for (d, *vs) in zip(data, *values.values())
        ),
    )
    return "\n".join(["```csv", *rows, "```"])


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
    """A class for representing a chunk."""

    id: THash = field()
    content: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)


# Graph types
@dataclass
class TEntity(BTResponseModel, BTNode):
    name: str = field()
    type: str = Field()
    description: str = Field()

    def to_str(self) -> str:
        return f"{self.name} ({self.type}): {self.description}"

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Entity"):
        name: str = Field(..., description="The name of the entity.")
        type: str = Field(..., description="The type of the entity.")
        description: str = Field(..., description="The description of the entity, in details and comprehensive.")

        @staticmethod
        def to_dataclass(pydantic: "TEntity.Model") -> "TEntity":
            return TEntity(name=pydantic.name, type=pydantic.type, description=pydantic.description)

        @field_validator("name", mode="before")
        @classmethod
        def uppercase_name(cls, value: str):
            return value.upper() if value else value

        @field_validator("type", mode="before")
        @classmethod
        def uppercase_type(cls, value: str):
            return value.upper() if value else value


class TQueryEntities(BaseModel):
    entities: List[str] = Field(
        ...,
        description=(
            "The named entities in the query. This list can be empty if there are no meaningful entities in the query."
        ),
    )
    n: int = Field(..., description="The number of named entities found.")  # So that the LLM can answer 0.

    @field_validator("entities", mode="before")
    @classmethod
    def uppercase_source(cls, value: List[str]):
        return [e.upper() for e in value] if value else value


@dataclass
class TRelation(BTResponseModel, BTEdge):
    source: str = field()
    target: str = field()
    description: str = field()
    chunks: Iterable[THash] | None = field(default=None)

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Relation"):
        source: str = Field(..., description="The name of the source entity.", alias="source_entity")
        target: str = Field(..., description="The name of the target entity.", alias="target_entity")
        description: str = Field(
            ..., description="The description of the relationship between the source and target entity"
        )

        @staticmethod
        def to_dataclass(pydantic: "TRelation.Model") -> "TRelation":
            return TRelation(source=pydantic.source, target=pydantic.target, description=pydantic.description)

        @field_validator("source", mode="before")
        @classmethod
        def uppercase_source(cls, value: str):
            return value.upper() if value else value

        @field_validator("target", mode="before")
        @classmethod
        def uppercase_target(cls, value: str):
            return value.upper() if value else value


@dataclass
class TGraph(BTResponseModel):
    entities: List[TEntity] = field()
    relationships: List[TRelation] = field()

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Graph"):
        entities: List[TEntity.Model] = Field(description="The entities in the graph.")
        relationships: List[TRelation.Model] = Field(description="The relationships between the entities.")

        @staticmethod
        def to_dataclass(pydantic: "TGraph.Model") -> "TGraph":
            return TGraph(
                entities=[p.to_dataclass(p) for p in pydantic.entities],
                relationships=[p.to_dataclass(p) for p in pydantic.relationships],
            )


class TEditRelation(BaseModel):
    ids: List[int] = Field(..., description="The ids of the facts that you are combining into one.")
    description: str = Field(
        ..., description="The summarized description of the combined facts, in detail and comprehensive."
    )


class TEditRelationList(BaseModel):
    groups: List[TEditRelation] = Field(
        ...,
        description="The list of new fact groups. Include only groups of more than one fact.",
        alias="grouped_facts",
    )


@dataclass
class TContext(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing the context used to generate a query response."""

    entities: List[GTNode]
    entity_scores: List[TScore]
    relationships: List[GTEdge]
    relationship_scores: List[TScore]
    chunks: List[GTChunk]
    chunk_scores: List[TScore]

    def to_str(self, max_len: Optional[int] = None) -> str:
        """Convert the context to a string representation."""
        if max_len:
            objects: List[List[Any]] = [self.chunks, self.entities, self.relationships]
            scores: List[List[TScore]] = [self.chunk_scores, self.entity_scores, self.relationship_scores]
            lengths = [
                [len(c.content) for c in self.chunks],
                [len(e.name) + len(e.description) for e in self.entities],
                [len(r.source) + len(r.target) + len(r.description) for r in self.relationships],
            ]
            # Compute entities len
            # Compute the total length
            total_len = sum(np.sum(v) for v in lengths)

            if total_len > max_len:
                # Remove elements based on their relative relevance
                # Compute the relative relevance

                while total_len > max_len:
                    lowest_relevances = [relevance[-1] if len(relevance) else TScore(2) for relevance in scores]
                    to_remove = np.argmin(lowest_relevances)

                    if lowest_relevances[to_remove] == 2:
                        break

                    if len(objects[to_remove]) > 0:
                        total_len -= lengths[to_remove].pop()
                        objects[to_remove].pop()
                        scores[to_remove].pop()

        data: List[str] = []
        if len(self.entities):
            data.extend(
                [
                    "#Entities",
                    dump_to_csv(self.entities, ["name", "description"], with_header=True),
                    "\n",
                ]
            )
        else:
            data.append("#Entities: None\n")

        if len(self.relationships):
            data.extend(
                [
                    "#Relationships",
                    dump_to_csv(self.relationships, ["source", "target", "description"], with_header=True),
                    "\n",
                ]
            )
        else:
            data.append("#Relationships: None\n")

        if len(self.chunks):
            data.extend(
                [
                    "#Sources",
                    dump_to_csv(self.chunks, ["content"], with_header=True, with_index=True),
                    "\n",
                ]
            )
        else:
            data.append("#Sources: None\n")
        context = "\n".join(data)

        return context


@dataclass
class TQueryResponse(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing a query response."""

    response: str
    context: TContext[GTNode, GTEdge, GTHash, GTChunk]
