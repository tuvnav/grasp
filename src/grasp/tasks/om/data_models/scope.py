import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp.tasks.om.data_models.patterns import PredicatePattern
from grasp.tasks.om.data_models.profile import GroupDefinition, RoleName
from grasp.tasks.om.foundation.text_normalization import (
    NORMALIZATION_LEVELS,
    STANDARD_LEVELS,
)
from grasp.tasks.om.foundation.types import Side

# the entities of a side that are not matched yet
REST = "rest"


# a group of the profile by name, REST, or a definition for this strategy only
type GroupRef = str | GroupDefinition


class Population(BaseModel):
    """Source and target entities a strategy compares."""
    model_config = ConfigDict(extra="forbid")
    source: list[GroupRef] = Field(default=[REST], min_length=1)
    target: list[GroupRef] = Field(default=[REST], min_length=1)


# values of an entity: one or more roles of its (effective) profile, or an own
# pattern over ?s (the entity) and ?o (the value); a pattern may compute ?o (BIND)
type ValuePatternSource = list[RoleName] | PredicatePattern


class ValuePatterns(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: ValuePatternSource = ["names", "synonyms"]
    target: ValuePatternSource | None = None  # None: like the source

    def side(self, side: Side) -> ValuePatternSource:
        return self.source if side == Side.SOURCE or self.target is None else self.target


# levels of the anchor normalization ladder (foundation/text_normalization.py), fine to coarse
type LevelName = Literal[*NORMALIZATION_LEVELS]


class Transform(BaseModel):
    """A string transform applied to every value before the normalization
    ladder. Only what the ladder cannot do (case, word splitting, word sets,
    plural and inflection are levels)."""
    model_config = ConfigDict(extra="forbid")
    op: Literal[
        "strip_accents",    # é -> e
        "remove_brackets",  # drop parenthesized parts: "Enterprise (NCC-1701)" -> "Enterprise"
        "regex_replace",    # re.sub(pattern, replacement)
        "digits_only",      # keep digits: "NCC-1701-D" -> "1701"
        "year",             # first four-digit year, or nothing
        "initials",         # "John Ronald Smith" -> "J R Smith"
        "transliterate",    # to Latin script (e.g. Cyrillic, Greek)
    ]
    pattern: str | None = None
    replacement: str = ""

    @model_validator(mode="after")
    def check_pattern(self) -> Self:
        if self.op == "regex_replace":
            if not self.pattern:
                raise ValueError("regex_replace needs a pattern")
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}") from e
        elif self.pattern is not None:
            raise ValueError(f"{self.op} takes no pattern")
        return self


class Normalization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transforms: list[Transform] = []
    # levels tried from fine to coarse (the anchor rule records the finest level
    # on which a pair is unique); a single level for one fixed normalization
    levels: list[LevelName] = Field(default=list(STANDARD_LEVELS), min_length=1)

    @model_validator(mode="after")
    def levels_in_ladder_order(self) -> Self:
        order = [level for level in NORMALIZATION_LEVELS if level in self.levels]
        if order != self.levels:
            raise ValueError(f"levels must be distinct and in ladder order: {', '.join(NORMALIZATION_LEVELS)}")
        return self
