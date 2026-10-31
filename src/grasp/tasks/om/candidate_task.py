from grasp.tasks.om.alignment_state import AlignmentState
from grasp.tasks.om.candidate_model import Generator
from grasp.tasks.om.model import Side
from grasp.tasks.om.profile_model import GroupDefinition, KgProfile
from grasp.tasks.om.scope_model import GroupRef, Population
from grasp.utils import FunctionCallException


def resolve_group_refs(kg_profile: KgProfile, group_refs: list[GroupRef]) -> list[GroupDefinition]:
    res = []
    for gr in group_refs:
        if isinstance(gr, str):
            if gr not in kg_profile.groups:
                raise FunctionCallException(
                    f"{gr} is not a group name in {kg_profile.kg_name}. "
                    f"If you want to use an existing group, choose between {kg_profile.groups.keys()}, "
                    "or else if you want to define an own group, call the function "
                    "with a 'GroupDefinition'-object, defining a new group."
                    )
            res.append(kg_profile.groups[gr].definition)
        else:  # type: GroupDefinition
            res.append(gr)
    return res


def resolve_population(kg_profile: KgProfile, population: Population) -> dict[Side, list[GroupDefinition]]:
    return {
        Side.SOURCE: resolve_group_refs(kg_profile, population.source),
        Side.TARGET: resolve_group_refs(kg_profile, population.target)
    }


def get_key_candidates(alignment_state: AlignmentState, kg_profile: KgProfile, generator: Generator):
    groups = resolve_population(kg_profile, generator.population)
