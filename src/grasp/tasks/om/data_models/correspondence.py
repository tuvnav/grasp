"""
Correspondences between entities of two KGs, with their provenance.
"""
from pydantic import BaseModel

from grasp.tasks.entities import Entity
from grasp.tasks.om.data_models.relations import EQUIVALENCE


class Provenance(BaseModel):
    """Where a correspondence of the pipeline comes from."""
    stage: str                    # e.g. anchors, candidates
    round: int = 0                # round of the candidate stage's meta-loop
    stratum: str | None = None    # strategy and stratum it was found in
    rationale: str | None = None  # judge's reason, if judged one by one


class Correspondence(BaseModel):
    # TODO: Enxtendable to handle complex patterns
    # besides atomic components?
    """
    A singe correspondence between two entities
    from different ontologies.
    """
    entity_source: Entity
    entity_target: Entity

    # OAEI standard
    measure: float = 1.0
    # iri of a RelationSpec from the task's relation profile
    relation: str = EQUIVALENCE.iri
    # set by the pipeline stages; None for correspondences of the agentic search
    provenance: Provenance | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entity_source.identifier, self.entity_target.identifier, self.relation)


class PotentialCorrespondences(BaseModel):
    source_entity: Entity
    candidates: list[Entity]
