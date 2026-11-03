from grasp.manager import KgManager
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidate_generators import Generator
from grasp.tasks.om.data_models.profile import GroupDefinition, KgProfile
from grasp.tasks.om.data_models.scope import GroupRef, Population
from grasp.tasks.om.foundation.types import Side
from grasp.tasks.om.kg_operations.grouping import Grouping, profile_groups_to_grouping
from grasp.tasks.om.kg_operations.value_retrieval import retrieve_value_rows
from grasp.utils import FunctionCallException


def get_group_grouings(
        kg_profile_source: KgProfile,
        kg_profile_target: KgProfile,
        population: Population
        ) -> dict[Side, Grouping]:

    inline_defined_groups_source = [
        g for g in population.source if isinstance(g, GroupDefinition)
        ]
    inline_defined_groups_target = [
        g for g in population.target if isinstance(g, GroupDefinition)
    ]

    if kg_profile_source.large or kg_profile_target.large:
        ...
    else:
        group_grouping_source = profile_groups_to_grouping(
            kg_profile_source, inline_defined_groups_source
            )
        group_grouping_target = profile_groups_to_grouping(
            kg_profile_target, inline_defined_groups_target
        )

    return {
        Side.SOURCE: group_grouping_source,
        Side.TARGET: group_grouping_target
        }


def get_key_candidates(
        kg_manager_source: KgManager,
        kg_manager_target: KgManager,
        alignment_state: AlignmentState,
        generator: Generator
        ):
    group_groupings = get_group_grouings(
        alignment_state.source_kg_profile,
        alignment_state.target_kg_profile,
        generator.population
        )
    source_group_grouping = group_groupings[Side.SOURCE]
    target_group_grouping = group_groupings[Side.TARGET]
    source_group_value_rows = retrieve_value_rows(
        kg_manager_source,
        source_group_grouping,
        alignment_state.source_kg_profile.entity_space,
        generator.strategy.values,
        alignment_state.source_kg_profile.roles
    )