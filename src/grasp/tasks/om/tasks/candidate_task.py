from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidate_generators import Generator
from grasp.tasks.om.data_models.profile import GroupDefinition, KgProfile
from grasp.tasks.om.data_models.scope import GroupRef, Population
from grasp.tasks.om.foundation.types import Side
from grasp.utils import FunctionCallException


def resolve_group_refs(kg_profile: KgProfile, group_refs: list[GroupRef]) -> list[GroupDefinition]:
    res = []
    for gr in group_refs:
        if isinstance(gr, str):
            if gr not in kg_profile.group_names():
                raise FunctionCallException(
                    f"{gr} is not a group name in {kg_profile.kg_name}. "
                    f"If you want to use an existing group, choose between {kg_profile.group_names()}, "
                    "or else if you want to define an own group, call the function "
                    "with a 'GroupDefinition'-object, defining a new group."
                    )
            res.append(kg_profile.get_group(gr).definition)
        else:  # type: GroupDefinition
            if kg_profile.group_name_already_exists(gr.name):
                raise ValueError(
                    f"{gr.name} already exists in this knowledge graph."
                    f" Please choose a new name. Already existing: {kg_profile.group_names()}"
                )
            res.append(gr)
    return res


def resolve_population(kg_profile: KgProfile, population: Population) -> dict[Side, list[GroupDefinition]]:
    return {
        Side.SOURCE: resolve_group_refs(kg_profile, population.source),
        Side.TARGET: resolve_group_refs(kg_profile, population.target)
    }


def get_key_candidates(
        alignment_state: AlignmentState, kg_profile: KgProfile, generator: Generator
        ):
    groups = resolve_population(kg_profile, generator.population)
    source_groups, target_groups = groups[Side.SOURCE], groups[Side.TARGET]
