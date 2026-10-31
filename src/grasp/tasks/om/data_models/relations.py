"""
Relations a correspondence may express (equivalence, STROMA relations) and the relation profiles of a task.
"""
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from grasp.configs import GraspConfig
from grasp.tasks.om.data_models.patterns import RelationAlgebra


class RelationSpec(RelationAlgebra):
    """
    A relation that may hold between a source and a target entity, together
    with the mathematical properties the system uses for consistency checks.
    `iri` is the relation's identifier; for OAEI relations the Alignment API
    symbol (e.g. "=") is used so that exported alignments stay unchanged.
    `description` is shown to the LLM and should state the functional
    criterion of the relation, read from the source to the target entity:
    what a query for one entity should return in the other system.
    """
    iri: str
    description: str
    # relations that cannot hold for the same entity pair simultaneously
    excludes: list[str] = []
    # weaker relations that hold for an entity pair whenever this one holds
    # (e.g. skos:exactMatch implies skos:closeMatch)
    implies: list[str] = []

    @model_validator(mode="after")
    def check_symmetry(self) -> "RelationSpec":
        # a relation is symmetric iff it is its own inverse
        if self.symmetric:
            if self.inverse_of is None:
                self.inverse_of = self.iri
            elif self.inverse_of != self.iri:
                raise ValueError(
                    f"Relation {self.iri} is symmetric, so it must be its "
                    f"own inverse, but inverse_of is {self.inverse_of}"
                )
        elif self.inverse_of == self.iri:
            self.symmetric = True
        return self


class RelationProfile(BaseModel):
    """
    The set of relations the LLM may choose from within one alignment task.
    """
    name: str
    relations: list[RelationSpec]
    # alignment-level cardinality in Alignment API notation (11, 1?, ??, **);
    # informational for now, not enforced
    cardinality: str = "**"
    # whether an ontology may contain duplicate entities, so that several
    # entities related by a symmetric and transitive relation to the same
    # entity of the other ontology can all be correct
    duplicates_within: bool = False
    # optional explanation of how to decide between the relations, shown to
    # the LLM right after the relation descriptions (e.g. an order in which
    # the relations are to be checked)
    description: str | None = None

    @model_validator(mode="after")
    def check_references(self) -> "RelationProfile":
        iris = [r.iri for r in self.relations]
        if not iris:
            raise ValueError(f"Relation profile {self.name} has no relations")
        if len(iris) != len(set(iris)):
            raise ValueError(f"Relation profile {self.name} has duplicate relations")
        for r in self.relations:
            for ref in [r.inverse_of, *r.excludes, *r.implies]:
                if ref is not None and ref not in iris:
                    raise ValueError(
                        f"Relation {r.iri} in profile {self.name} references "
                        f"unknown relation {ref}"
                    )
        return self

    def get(self, iri: str) -> RelationSpec | None:
        return next((r for r in self.relations if r.iri == iri), None)


EQUIVALENCE = RelationSpec(
    iri="=",
    description="The source and the target entity refer to the same real-world things: "
    "a query for either entity should also return the entries of the other.",
    symmetric=True,
    transitive=True,
)


# relation types of the STROMA/TaSeR datasets, as used in the OAEI Beyond
# Equivalence track (MELT: TrackRepository.GeneralRelation)
STROMA_RELATIONS = [
    EQUIVALENCE.model_copy(update={"excludes": ["<", ">"]}),
    RelationSpec(
        iri="<",
        description="The source entity is narrower than the target entity: a query for the "
        "target entity should also return all entries of the source entity, but not vice "
        "versa (e.g. 'Jeans' < 'Clothing').",
        transitive=True,
        inverse_of=">",
        excludes=["=", ">"],
    ),
    RelationSpec(
        iri=">",
        description="The source entity is broader than the target entity: a query for the "
        "source entity should also return all entries of the target entity, but not vice "
        "versa (e.g. 'Clothing' > 'Jeans').",
        transitive=True,
        inverse_of="<",
        excludes=["=", "<"],
    ),
    RelationSpec(
        iri="HasA",
        description="Entries of the target entity are parts or components of entries of the "
        "source entity (e.g. 'Suit' HasA 'Trousers').",
        inverse_of="PartOf",
    ),
    RelationSpec(
        iri="PartOf",
        description="Entries of the source entity are parts or components of entries of the "
        "target entity (e.g. 'Trousers' PartOf 'Suit').",
        inverse_of="HasA",
    ),
    RelationSpec(
        iri="Related",
        description="The entities are closely associated, so a system using one of them "
        "should link to the other, but none of the other relations holds.",
        symmetric=True,
    ),
]


RELATION_PROFILES: dict[str, RelationProfile] = {
    "equivalence": RelationProfile(
        name="equivalence", relations=[EQUIVALENCE], cardinality="11"
    ),
    "stroma": RelationProfile(
        name="stroma", relations=STROMA_RELATIONS, cardinality="**"
    ),
}


def allows_many(profile: RelationProfile) -> bool:
    """
    Whether entities may take part in several correspondences, i.e. the
    profile's cardinality (Alignment API notation) contains '*' or '+'.
    """
    return any(c in profile.cardinality for c in "*+")


def load_relation_profile(spec: str | dict) -> RelationProfile:
    """
    `spec` is the name of a built-in profile, the path to a YAML or JSON
    file with a profile, or a profile given inline as a mapping.
    """
    if isinstance(spec, dict):
        return RelationProfile(**spec)
    if spec in RELATION_PROFILES:
        return RELATION_PROFILES[spec]
    path = Path(spec).expanduser()
    if path.suffix not in (".yaml", ".yml", ".json"):
        raise ValueError(
            f"Unknown relation profile {spec}, must be one of "
            f"{list(RELATION_PROFILES)} or a .yaml/.yml/.json file"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return RelationProfile(**data)


def get_relation_profile(config: GraspConfig) -> RelationProfile:
    return load_relation_profile(
        config.task_kwargs.get("om", {}).get("relation_profile", "equivalence")
    )
