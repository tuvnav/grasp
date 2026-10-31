"""
The five stages of the KG profile (profile_model.py), each a GRASP task on one KG:
    om-profile-space       a.  entity space
    om-profile-partitions  b1. exclusive partitions (kinds) and which do not compete
    om-profile-roles       c.  KG-wide roles and structure relations
    om-profile-groups      b2. overlapping groups, with the task in view; exclusion
    om-profile-group       d.  one group: local names and deviations from the KG-wide roles
"""
from collections import Counter
from enum import StrEnum
import re
from typing import Callable

from grasp.manager import KgManager
from grasp.tasks.om.kg_profile import (
    Coverage, Grouping, LocalNameStats, Sides, canon_roles, complement, default_entity_space,
    default_roles, distinguishing_features, element_sparql, element_usage, estimated_sizes, group_grouping,
    group_sizes, grouping_query, kind_partitions, local_name_stats, local_name_type, member_coverage,
    members_query, membership_overlap, partition_grouping, predicate_coverage, prefix_counts, prefix_totals,
    rest, sample_entities, type_coverage,
)
from grasp.tasks.om.kg_sample import (
    class_counts, count_triples, current_sample, default_class_groups, is_large, large_prefix_counts,
    type_membership, values_block,
)
from grasp.tasks.om.model import (
    LABEL_PREDICATES, RDF_TYPE, UNCATEGORIZED, VOCABULARY, NameType, configured_namespaces, namespace_of,
    sparql_path,
)
from grasp.tasks.om.profile_model import (
    ROLE_FIELDS, GroupDefinition, GroupOrigin, GroupProfile, KgProfile, Partition, Roles, Separation, Stage,
    derived_roles, describe_profile, diff_roles, effective_roles, role_elements, roles_from_elements, short_iri,
)
from grasp.tasks.om.profile_stage import (
    RoleEditingTask, ProfileStageTask, run_rows, share,
)
from grasp.tasks.om.render import clip
from grasp.tasks.om.retrieve import select_rows
from grasp.utils import FunctionCallException, format_section

NUM_NAMESPACES = 15
# kinds or classes listed per prefix
NUM_KINDS = 8
NUM_FEATURES = 10
NUM_PREDICATES = 25
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# stage a counts entities per prefix by kind, without the subjects outside all kinds
SPACE_GROUPING = partition_grouping(kind_partitions(), rest=False)
# name of all entities of a grouping, to compare a group with the rest
ALL = "_all"


def entity_examples(task: ProfileStageTask, grouping: Grouping, space: list[str], where: str, k: int) -> list[str]:
    """Up to k entities with a name and their partition or group."""
    names = sparql_path(LABEL_PREDICATES)
    rows = run_rows(task.manager, grouping_query(
        task.manager, grouping, space, "?entity ?group ?name",
        f"{where}\nOPTIONAL {{ ?entity {names} ?name }}", f"LIMIT {k * 4}",
    ))
    seen, lines = set(), []
    for row in rows:
        entity = row["entity"].value
        if entity in seen:
            continue
        seen.add(entity)
        name = f' "{clip(row["name"].value, 50)}"' if "name" in row else ""
        lines.append(f"  {entity}{name} ({row['group'].value})")
        if len(lines) == k:
            break
    return lines


def ensure_space(task: ProfileStageTask) -> list[str]:
    if task.input_profile.entity_space:
        return list(task.input_profile.entity_space)
    space, _ = default_entity_space(task.manager, task.config)
    return space


def default_partitions(task: ProfileStageTask, space: list[str]) -> dict[str, Partition]:
    """The kind partitions with entities (in a large KG without the alternatives
    that cannot be evaluated there)."""
    large = task.input_profile.large
    partitions = kind_partitions(large=large)
    sizes = estimated_sizes(task.manager, partition_grouping(partitions, rest=False), space)
    return {name: p for name, p in partitions.items() if sizes.get(name)}


def ensure_partitions(task: ProfileStageTask, space: list[str]) -> dict[str, Partition]:
    if task.input_profile.partitions:
        return dict(task.input_profile.partitions)
    return default_partitions(task, space)


def default_separations(partitions: dict[str, Partition]) -> list[Separation]:
    """Properties do not compete with the other kinds (the default before stage b1 decides)."""
    if "property" not in partitions:
        return []
    return [Separation(a="property", b=name, reason="properties and entities are not equivalent")
            for name in partitions if name != "property"]


def ensure_groups(task: ProfileStageTask, space: list[str]) -> dict[str, GroupDefinition]:
    """The groups of the input profile; otherwise one per partition and, in a
    large KG, one per large class."""
    profile = task.input_profile
    if profile.groups:
        return {name: g.definition for name, g in profile.groups.items()}
    groups = {name: GroupDefinition(membership_sparql=p.membership_sparql, origin=GroupOrigin.PARTITION,
                                    features=p.features)
              for name, p in ensure_partitions(task, space).items()}
    if profile.large:
        for name, type_iri, _ in default_class_groups(task.manager, tuple(space), class_counts(task.manager)):
            if name not in groups:
                groups[name] = GroupDefinition(membership_sparql=type_membership(type_iri), origin=GroupOrigin.CLASS,
                                               features=f"typed {short_iri(type_iri)}")
    return groups


# --- a. entity space ----------------------------------------------------------------

class SpaceTask(ProfileStageTask):
    name = "om-profile-space"
    stage = Stage.SPACE
    prompt_file = "profile_space.txt"

    def prepare(self, input: dict) -> str:
        self.triples = count_triples(self.manager)
        self.large = is_large(self.triples)
        if self.large:
            # a full count per prefix does not finish: estimate it from class samples
            self.classes = class_counts(self.manager)
            prefix_counts_ = large_prefix_counts(self.manager, self.classes)
            all_counts, named = prefix_counts_.counts, prefix_counts_.named
            named_totals, totals = prefix_counts_.named_totals, prefix_counts_.totals
            self.prefix_examples = prefix_counts_.examples
            self.class_prefix_counts = all_counts
        else:
            named, all_counts = prefix_counts(self.manager, True), prefix_counts(self.manager, False)
            named_totals, totals = prefix_totals(named), prefix_totals(all_counts)
        configured = configured_namespaces(self.config, self.manager.kg)
        if self.input_profile.entity_space:
            space, source = list(self.input_profile.entity_space), "given in the input profile"
        elif configured:
            space, source = list(configured), "configured"
        elif named_totals:
            space, source = [named_totals.most_common(1)[0][0]], "prefix with the most named entities"
        elif totals:
            space, source = [totals.most_common(1)[0][0]], "prefix with the most entities"
        else:
            space, source = [], "no entities found"
        self.stats = {"triples": self.triples, "large": self.large, "prefixes": dict(totals.most_common()),
                      "named": dict(named_totals.most_common())}
        self.default = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples, "large": self.large}, deep=True)
        hosts: dict[str, list[str]] = {}
        for prefix, _ in totals.most_common():
            hosts.setdefault(namespace_of(prefix), []).append(prefix)
        approx = "~" if self.large else ""
        lines = []
        for host, prefixes in sorted(hosts.items(), key=lambda x: -sum(totals[p] for p in x[1]))[:NUM_NAMESPACES]:
            if prefixes != [host]:
                lines.append(f"  {host}: {approx}{sum(totals[p] for p in prefixes)} entities, "
                             f"{approx}{sum(named_totals.get(p, 0) for p in prefixes)} with a canon name")
            for prefix in prefixes[:NUM_NAMESPACES]:
                kinds = sorted(((k, n) for (other, k), n in all_counts.items() if other == prefix), key=lambda x: -x[1])
                more = f", ... {len(kinds) - NUM_KINDS} more" if len(kinds) > NUM_KINDS else ""
                kinds_text = ", ".join(f"{k} {approx}{n}" for k, n in kinds[:NUM_KINDS]) + more
                indent = "    " if prefixes != [host] else "  "
                lines.append(f"{indent}{prefix}: {approx}{totals[prefix]} entities ({kinds_text}), "
                             f"{approx}{named_totals.get(prefix, 0)} with a canon name")
            if len(prefixes) > NUM_NAMESPACES:
                lines.append(f"    ... {len(prefixes) - NUM_NAMESPACES} more prefixes")
        if len(hosts) > NUM_NAMESPACES:
            lines.append(f"  ... {len(hosts) - NUM_NAMESPACES} more hosts")
        examples = []
        for prefix, _ in totals.most_common(3):
            examples.append(f"{prefix}:")
            examples += (self.sampled_examples(prefix, 4) if self.large
                         else entity_examples(self, SPACE_GROUPING, [prefix], "", 4))
        size = (f"Size: {self.triples} triples." if self.triples is not None
                else "Size: the triples cannot be counted (the query times out).")
        if self.large:
            size += (" This is a large knowledge graph: entities are counted per class (rdf:type) and "
                     "estimated from samples of each class (~); untyped entities are not counted.")
            group = "class (rdf:type)"
        else:
            group = "kind"
        return "\n\n".join([
            size,
            f"Default entity space ({source}): " + (", ".join(space) or "none"),
            f"Entities per IRI prefix (host, and below it the first path segment) and {group}:\n" + "\n".join(lines),
            "Examples:\n" + "\n".join(examples),
        ])

    def sampled_examples(self, prefix: str, k: int) -> list[str]:
        """Sampled entities under a prefix (large KG) with a name and their class."""
        pairs = [(e, c) for p, sampled in self.prefix_examples.items() if p.startswith(prefix) or prefix.startswith(p)
                 for e, c in sampled if e.startswith(prefix)][:k]
        if not pairs:
            return []
        names = {}
        rows = run_rows(self.manager, f"SELECT ?entity ?name WHERE {{ {values_block([e for e, _ in pairs])} "
                                      f"?entity {sparql_path(LABEL_PREDICATES)} ?name }}")
        for row in rows:
            names.setdefault(row["entity"].value, row["name"].value)
        return [f"  {e}" + (f' "{clip(names[e], 50)}"' if e in names else "") + f" ({c})" for e, c in pairs]

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [({
            "name": "inspect_namespace",
            "description": ("Shows the entities under an IRI prefix: their number per kind, how many have a "
                            "canon name, and examples with names and types."),
            "parameters": {
                "type": "object",
                "properties": {"prefix": {"type": "string", "description": "IRI prefix, e.g. http://example.org/onto#"}},
                "required": ["prefix"],
                "additionalProperties": False,
            },
        }, self.inspect_namespace)]

    def finish_properties(self) -> dict[str, dict]:
        return {"entity_space": {
            "type": "array", "items": {"type": "string"},
            "description": "IRI prefixes of the entities to align.",
        }}

    def finish_required(self) -> list[str]:
        return ["entity_space"]

    def inspect_namespace(self, fn_args: dict) -> str:
        prefix = fn_args.get("prefix") or ""
        if not prefix:
            raise FunctionCallException("prefix required")
        if self.large:
            return self.inspect_sampled(prefix)
        sizes = run_sizes(self, SPACE_GROUPING, [prefix])
        if not sizes:
            return f"No entities with IRIs starting with {prefix}."
        named = run_rows(self.manager, grouping_query(
            self.manager, SPACE_GROUPING, [prefix], "(COUNT(DISTINCT ?entity) AS ?n)",
            f"FILTER EXISTS {{ ?entity {sparql_path(LABEL_PREDICATES)} ?name }}",
        ))
        n_named = int(named[0]["n"].value) if named and "n" in named[0] else 0
        types = run_rows(self.manager, grouping_query(
            self.manager, SPACE_GROUPING, [prefix], "?t (COUNT(DISTINCT ?entity) AS ?n)",
            f"?entity <{RDF_TYPE}> ?t", "GROUP BY ?t ORDER BY DESC(?n) LIMIT 8",
        ))
        return "\n".join([
            f"{prefix}: " + ", ".join(f"{p} {n}" for p, n in sizes.items()) + f"; {n_named} with a canon name",
            "Most frequent types: " + (", ".join(f"{short_iri(r['t'].value)} {r['n'].value}" for r in types if "t" in r) or "none"),
            "Examples:", *entity_examples(self, SPACE_GROUPING, [prefix], "", 8),
        ])

    def inspect_sampled(self, prefix: str) -> str:
        """inspect_namespace of a large KG, from the class samples."""
        classes = Counter()
        for (p, c), n in self.class_prefix_counts.items():
            if p.startswith(prefix) or prefix.startswith(p):
                classes[c] += n
        examples = self.sampled_examples(prefix, 8)
        if not examples:
            return (f"No sampled entity has an IRI starting with {prefix}. (In this large knowledge graph, "
                    "entities are only sampled per class; untyped entities are not seen.)")
        return "\n".join([
            f"{prefix}: classes of the sampled entities under this prefix or its namespace (estimated members): "
            + ", ".join(f"{c} ~{n}" for c, n in classes.most_common(NUM_NAMESPACES)),
            "Examples:", *examples,
        ])

    def has_entities(self, prefix: str) -> bool:
        if not self.large:
            return bool(run_sizes(self, SPACE_GROUPING, [prefix]))
        if any(e.startswith(prefix) for sampled in self.prefix_examples.values() for e, _ in sampled):
            return True
        try:
            return bool(select_rows(self.manager, f'SELECT ?s WHERE {{ ?s ?p ?o FILTER(STRSTARTS(STR(?s), "{prefix}")) }} LIMIT 1'))
        except Exception:
            return True  # cannot tell in time; trust the model

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        space = [ns for ns in fn_args.get("entity_space") or [] if ns]
        if not space:
            raise FunctionCallException("entity_space must name at least one IRI prefix")
        empty = [ns for ns in space if not self.has_entities(ns)]
        if empty:
            raise FunctionCallException(f"No entities under {empty}")
        profile = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples, "large": self.large}, deep=True)
        return profile, f"Entity space: {', '.join(space)}"


def size_text(task, name: str) -> str:
    """Size of a partition or group: counted, or in a large KG estimated with the size of its sample."""
    if current_sample(task.manager) is None:
        return f"{task.sizes.get(name, 0)} entities"
    estimate = task.estimates.get(name)
    return f"{'~' + str(estimate) if estimate is not None else 'unknown number of'} entities, sample {task.sizes.get(name, 0)}"


def size_stats(task) -> dict:
    """Sizes for the output: estimates in a large KG, with the sample sizes."""
    if current_sample(task.manager) is None:
        return {"sizes": task.sizes}
    return {"sizes": task.estimates, "sample_sizes": task.sizes}


def run_sizes(task: ProfileStageTask, grouping: Grouping, space: list[str]) -> dict[str, int]:
    try:
        return group_sizes(task.manager, grouping, space)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query failed: {e}") from e


def format_features(inside: dict[str, int], size_inside: int, outside: dict[str, int], size_outside: int,
                    other: str) -> list[str]:
    return [f"  {short_iri(f.key)}: {f.inside:.0%} inside, {f.outside:.0%} in {other}"
            for f in distinguishing_features(inside, size_inside, outside, size_outside, NUM_FEATURES)]


def membership_argument(fn_args: dict) -> str:
    membership = (fn_args.get("membership") or "").strip()
    if not re.search(r"\?entity\b", membership):
        raise FunctionCallException("The membership pattern must use ?entity")
    return membership


def valid_name(name: str, unit: str) -> str:
    if not NAME.match(name or "") or name in (UNCATEGORIZED, ALL):
        raise FunctionCallException(f"Invalid {unit} name {name!r}: lower case letters, digits and _")
    return name


class GroupingStageTask(ProfileStageTask):
    """Stages b1 and b2: definitions of partitions or groups by membership
    patterns, tested before they are set, with sizes and distinguishing
    features. Subclasses say how the definitions form a Grouping."""
    unit: str  # "partition" or "group", in texts for the model
    space: list[str]
    definitions: dict  # name -> Partition or GroupDefinition, in order
    tested: set[str]
    sizes: dict[str, int]
    estimates: dict[str, int]
    predicates: dict[str, dict[str, int]]
    types: dict[str, dict[str, int]]

    def grouping(self) -> Grouping:
        raise NotImplementedError

    def names(self) -> list[str]:
        raise NotImplementedError

    def outside(self, kind: str, name: str) -> tuple[dict[str, int], int]:
        """Counts of predicates or types (kind) and size of the entities outside one partition or group."""
        raise NotImplementedError

    def update_statistics(self) -> None:
        grouping = self.grouping()
        try:
            self.sizes = group_sizes(self.manager, grouping, self.space)
            self.estimates = estimated_sizes(self.manager, grouping, self.space)
            self.predicates = {p: {k: c.entities for k, c in counts.items()}
                               for p, counts in predicate_coverage(self.manager, grouping, self.space).items()}
            self.types = type_coverage(self.manager, grouping, self.space)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e

    def describe_member(self, name: str, title: str) -> str:
        size = self.sizes.get(name, 0)
        lines = [f"{title} ({size_text(self, name)}). Examples:",
                 *entity_examples(self, self.grouping(), self.space, f'FILTER(?group = "{name}")', 3)]
        outside, size_outside = self.outside("predicates", name)
        lines.append("Predicates (share of entities with the predicate):")
        lines += format_features(self.predicates.get(name, {}), size, outside, size_outside, f"other {self.unit}s")
        if self.types.get(name):
            outside, size_outside = self.outside("types", name)
            lines.append("Types:")
            lines += format_features(self.types[name], size, outside, size_outside, f"other {self.unit}s")[:5]
        return "\n".join(lines)

    def update_definitions(self, definitions: dict) -> None:
        """Sets the definitions (self.definitions) and their statistics; on a failed query, nothing changes."""
        old = self.definitions
        self.definitions = definitions
        try:
            self.update_statistics()
        except FunctionCallException:
            self.definitions = old
            raise

    def known_name(self, name: str | None) -> str:
        if name not in self.names():
            raise FunctionCallException(f"Unknown {self.unit} {name}; {self.unit}s: {', '.join(self.names())}")
        return name

    def test_definition(self, fn_args: dict) -> str:
        membership = membership_argument(fn_args)
        try:
            size, overlap = membership_overlap(self.manager, self.grouping(), self.space, membership)
            inside, inside_types = member_coverage(self.manager, self.space, membership)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        self.tested.add(membership)
        if not size:
            return "No entity of the entity space matches."
        sample = current_sample(self.manager)
        if sample is not None:
            matches = sample.sample(membership)
            estimate = (f"~{round(matches.count * matches.space_share)}" if matches.count is not None
                        else "an unknown number of")
            size_line = f"{estimate} entities of the entity space match (statistics on a sample of {size}); "
        else:
            size_line = f"{size} entities match; "
        compare = fn_args.get("compare_with")
        if compare is not None:
            self.known_name(compare)
        else:
            compare = max((p for p in overlap if p in self.sizes), key=overlap.get, default=None)
        where = f"by current {self.unit} (share of the {self.unit} that matches)"
        lines = [size_line + where + ": "
                 + ", ".join(f"{p} {share(n, self.sizes.get(p, 0))}" for p, n in sorted(overlap.items(), key=lambda x: -x[1]))]
        if compare is not None:
            rest_size = self.sizes.get(compare, 0) - overlap.get(compare, 0)
            for title, counts, ins in (("Predicates", self.predicates, inside), ("Types", self.types, inside_types)):
                outside = {k: max(0, n - ins.get(k, 0)) for k, n in counts.get(compare, {}).items()}
                lines.append(f"{title} (share of the matches vs. the rest of {compare}):")
                lines += format_features(ins, size, outside, rest_size, f"rest of {compare}")[:NUM_FEATURES if title == "Predicates" else 5]
        lines.append("Examples:")
        names = sparql_path(LABEL_PREDICATES)
        rows = run_rows(self.manager, (
            f"SELECT DISTINCT ?entity ?name WHERE {{ {members_query(membership, self.space, self.manager)} "
            f"OPTIONAL {{ ?entity {names} ?name }} }} LIMIT 6"
        ))
        lines += [f"  {r['entity'].value}" + (f' "{clip(r["name"].value, 50)}"' if "name" in r else "")
                  for r in rows if "entity" in r]
        return "\n".join(lines)

    def checked_membership(self, fn_args: dict) -> str:
        membership = membership_argument(fn_args)
        if membership not in self.tested:
            raise FunctionCallException(f"Test the membership pattern with test_{self.unit} first (exactly this pattern)")
        return membership


def test_function(unit: str, description: str) -> dict:
    return {
        "name": f"test_{unit}",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"membership": MEMBERSHIP,
                           "compare_with": {"type": "string", "description": f"The {unit} to compare with."}},
            "required": ["membership"],
            "additionalProperties": False,
        },
    }


MEMBERSHIP = {"type": "string", "description": (
    "SPARQL group pattern over ?entity, e.g. ?entity <http://example.org/p> ?v")}


# --- b1. partitions ----------------------------------------------------------------

class PartitionsTask(GroupingStageTask):
    name = "om-profile-partitions"
    stage = Stage.PARTITIONS
    prompt_file = "profile_partitions.txt"
    unit = "partition"

    def prepare(self, input: dict) -> str:
        self.space = ensure_space(self)
        self.definitions = ensure_partitions(self, self.space)
        self.separations = (list(self.input_profile.separations) if self.input_profile.partitions
                            else default_separations(self.definitions))
        self.tested = set()
        self.update_statistics()
        self.stats = size_stats(self)
        self.default = self.with_partitions()
        sections = [self.format_sizes(), self.format_separations()]
        for name in self.names() + [UNCATEGORIZED]:
            if self.sizes.get(name):
                title = (f"Partition {name}" if name != UNCATEGORIZED
                         else f"Subjects that match no partition, {UNCATEGORIZED}, not aligned")
                sections.append(self.describe_member(name, title))
        return "\n\n".join(sections)

    def grouping(self) -> Grouping:
        # the subjects outside all partitions only in a small KG (a large one is not scanned)
        return partition_grouping(self.definitions, rest=not self.input_profile.large)

    def names(self) -> list[str]:
        return list(self.definitions)

    def outside(self, kind: str, name: str) -> tuple[dict[str, int], int]:
        return complement(getattr(self, kind), self.sizes, name)

    def format_sizes(self) -> str:
        lines = [f"  {name}: {size_text(self, name)}: {{ {d.membership_sparql} }}"
                 + (f"; {d.features}" if d.features else "") for name, d in self.definitions.items()]
        if self.sizes.get(UNCATEGORIZED):
            lines.append(f"  {UNCATEGORIZED}: {self.sizes[UNCATEGORIZED]} subjects match no partition")
        return "Partitions in order (an entity belongs to the first that matches):\n" + "\n".join(lines)

    def format_separations(self) -> str:
        if not self.separations:
            return "Partitions that do not compete with each other: none (all compete)."
        return "Partitions that do not compete with each other:\n" + "\n".join(
            f"  {s.a} - {s.b}" + (f": {s.reason}" if s.reason else "") for s in self.separations)

    def with_partitions(self) -> KgProfile:
        """The input profile with the partitions that have entities and their separations."""
        kept = {name: d for name, d in self.definitions.items() if self.sizes.get(name)}
        separations = [s for s in self.separations if s.pair() <= set(kept)]
        return self.input_profile.model_copy(update={
            "entity_space": self.space, "partitions": kept, "separations": separations}, deep=True)

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_function("partition", (
                "Tests a membership pattern: how many entities of the entity space match, in which "
                "current partitions they are, which predicates and types distinguish them from the "
                "rest of the partition they overlap most (or compare_with), and examples. A pattern "
                "must be tested before it can be set.")), self.test_definition),
            ({
                "name": "set_partition",
                "description": (
                    "Defines a partition by a tested membership pattern, or redefines an existing one "
                    "(it keeps its place). A new partition is placed before the partition named in "
                    "before, otherwise first. Shows the partition sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the kind."},
                        "membership": MEMBERSHIP,
                        "features": {"type": "string", "description": "What distinguishes its entities."},
                        "before": {"type": "string"},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_partition),
            ({
                "name": "remove_partition",
                "description": "Removes a partition; its entities fall to the next matching partition.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_partition),
            ({
                "name": "set_competition",
                "description": (
                    "Sets whether the entities of two partitions compete with each other: if not, an "
                    "entity of one is never a candidate or a rival for an entity of the other."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"}, "b": {"type": "string"},
                        "compete": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["a", "b", "compete", "reason"],
                    "additionalProperties": False,
                },
            }, self.set_competition),
        ]

    def set_partition(self, fn_args: dict) -> str:
        name = valid_name(fn_args.get("name"), self.unit)
        membership = self.checked_membership(fn_args)
        before = fn_args.get("before")
        if before is not None and self.known_name(before) == name:
            raise FunctionCallException("A partition cannot be placed before itself")
        definition = Partition(membership_sparql=membership, features=fn_args.get("features") or "")
        items = [(n, d) for n, d in self.definitions.items() if n != name]
        if name in self.definitions:
            index = list(self.definitions).index(name)
        elif before is not None:
            index = [n for n, _ in items].index(before)
        else:
            index = 0
        items.insert(index, (name, definition))
        self.update_definitions(dict(items))
        return f"Partition {name} set.\n{self.format_sizes()}"

    def remove_partition(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last partition cannot be removed")
        self.update_definitions({n: d for n, d in self.definitions.items() if n != name})
        self.separations = [s for s in self.separations if name not in s.pair()]
        return f"Partition {name} removed.\n{self.format_sizes()}"

    def set_competition(self, fn_args: dict) -> str:
        a, b = self.known_name(fn_args.get("a")), self.known_name(fn_args.get("b"))
        if a == b:
            raise FunctionCallException("Entities of one partition always compete with each other")
        self.separations = [s for s in self.separations if s.pair() != {a, b}]
        if not fn_args.get("compete"):
            self.separations.append(Separation(a=a, b=b, reason=fn_args.get("reason") or ""))
        return self.format_separations()

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        profile = self.with_partitions()
        if not profile.partitions:
            raise FunctionCallException("No partition has entities")
        self.stats = size_stats(self)
        dropped = [name for name in self.definitions if name not in profile.partitions]
        summary = "Partitions: " + ", ".join(f"{n} {self.stats['sizes'].get(n, 0)}" for n in profile.partitions)
        if dropped:
            summary += f"; dropped as empty: {', '.join(dropped)}"
        return profile, summary


# --- c. KG-wide roles ----------------------------------------------------------------

def format_values(values: int, literals: int, blanks: int) -> str:
    """Kinds of the values (objects) of a predicate."""
    if not values:
        return ""
    iris = values - literals - blanks
    parts = [f"{iris / values:.0%} IRI"]
    if blanks:
        parts.append(f"{blanks / values:.0%} blank node")
    parts.append(f"{literals / values:.0%} literal")
    return ", ".join(parts)


def format_predicate_overview(coverage: dict[str, dict[str, Coverage]], sizes: dict[str, int],
                              members: list[str], roles: Roles, k: int) -> str:
    """The predicates with the highest coverage in some partition: the share of
    each partition's entities that are its subject (as ?s), the kinds of its
    values (?o) over these partitions, and the role it has."""
    in_role = {key[2]: key[0] for key in role_elements(roles) if key[1] == "predicate"}
    best: dict[str, float] = {}
    for p in members:
        for predicate, c in coverage.get(p, {}).items():
            if sizes.get(p):
                best[predicate] = max(best.get(predicate, 0.0), c.entities / sizes[p])
    lines = []
    for predicate in sorted(best, key=lambda x: -best[x])[:k]:
        per_member = []
        values = literals = blanks = 0
        for p in members:
            c = coverage.get(p, {}).get(predicate)
            if c is not None:
                per_member.append(f"{p} {c.entities / sizes[p]:.0%}")
                values, literals, blanks = values + c.values, literals + c.literals, blanks + c.blanks
        role = f" [{in_role[predicate]}]" if predicate in in_role else ""
        lines.append(f"  {short_iri(predicate)}{role}  as ?s: {', '.join(per_member)}"
                     f"  |  values: {format_values(values, literals, blanks)}")
    return "\n".join(lines)


class RolesTask(RoleEditingTask):
    name = "om-profile-roles"
    stage = Stage.ROLES
    prompt_file = "profile_roles.txt"
    unit = "partition"

    def prepare(self, input: dict) -> str:
        self.space = ensure_space(self)
        profile = self.input_profile
        if not profile.partitions:
            profile = profile.model_copy(update={"partitions": ensure_partitions(self, self.space)})
        self.grouping = partition_grouping(profile.partitions, rest=False)
        self.scope = list(profile.partitions)
        self.sizes = group_sizes(self.manager, self.grouping, self.space)
        self.estimates = estimated_sizes(self.manager, self.grouping, self.space)
        sides: dict[str, Sides] = {}
        usage = element_usage(self.manager, self.grouping, self.space, canon_roles(), sides)
        default = default_roles(usage, self.sizes, self.scope)
        self.init_roles(default, usage, sides, canon_roles())
        coverage = predicate_coverage(self.manager, self.grouping, self.space)
        self.stats = size_stats(self)
        self.default = profile.model_copy(update={"entity_space": self.space, "roles": default}, deep=True)
        canon = ", ".join(short_iri(key[2]) if key[1] == "predicate" else key[2]
                          for key in role_elements(canon_roles()))
        return "\n\n".join([
            "Partitions: " + ", ".join(f"{p} ({size_text(self, p)})" for p in self.scope),
            f"Default canon checked: {canon}. Canon elements below "
            "1% coverage in every partition are left out of the default.",
            "Default roles (coverage as ?s per partition):\n" + self.format_roles(default),
            "Most frequent predicates: share of each partition's entities that are their subject "
            "(as ?s), kinds of their values (?o), [role] if in the default:\n"
            + format_predicate_overview(coverage, self.sizes, self.scope, default, NUM_PREDICATES),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_functions()

    def stage_rules(self) -> list[str]:
        return ["Test every predicate or pattern you add with try_pattern first."]

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.checked_roles()
        profile = self.default.model_copy(update={"roles": roles}, deep=True)
        changed = [f for f in (*ROLE_FIELDS, "structure") if getattr(roles, f) != getattr(self.default.roles, f)]
        return profile, "Changed: " + (", ".join(changed) or "nothing (default roles)")


# --- b2. groups ----------------------------------------------------------------------

# exclude_group warns if the excluded groups cover more of the entity space
LARGE_EXCLUSION_SHARE = 0.5


class GroupsTask(GroupingStageTask):
    """Input besides kg and profile: partner_profile, the profile of the other
    KG of the task (optional; shown to the model if given)."""
    name = "om-profile-groups"
    stage = Stage.GROUPS
    prompt_file = "profile_groups.txt"
    unit = "group"

    def prepare(self, input: dict) -> str:
        self.space = ensure_space(self)
        self.definitions = ensure_groups(self, self.space)
        self.tested = set()
        self.update_statistics()
        self.stats = size_stats(self)
        self.default = self.with_groups()
        sections = []
        partner = input.get("partner_profile")
        if partner:
            notes = describe_profile(KgProfile.model_validate(partner))
            sections.append(format_section("Profile of the other knowledge graph of the task",
                                           "\n".join(f"- {n}" for n in notes)))
        sections.append(self.format_sizes())
        for name in self.names():
            if self.sizes.get(name):
                sections.append(self.describe_member(name, f"Group {name}"))
        return "\n\n".join(sections)

    def grouping(self) -> Grouping:
        return group_grouping(self.definitions)

    def names(self) -> list[str]:
        return list(self.definitions)

    def update_statistics(self) -> None:
        super().update_statistics()
        # all entities in any group, to compare a group with the rest
        merged = self.grouping().merged(ALL)
        try:
            self.all_size = group_sizes(self.manager, merged, self.space).get(ALL, 0)
            self.all_predicates = {p: c.entities for p, c in
                                   predicate_coverage(self.manager, merged, self.space).get(ALL, {}).items()}
            self.all_types = type_coverage(self.manager, merged, self.space).get(ALL, {})
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e

    def outside(self, kind: str, name: str) -> tuple[dict[str, int], int]:
        total = self.all_predicates if kind == "predicates" else self.all_types
        return rest(total, self.all_size, getattr(self, kind).get(name, {}), self.sizes.get(name, 0))

    def format_sizes(self) -> str:
        lines = []
        for name, d in self.definitions.items():
            notes = [d.origin.value] + ([f"part of {d.parent}"] if d.parent else [])
            excluded = " [excluded]" if d.excluded else ""
            lines.append(f"  {name}: {size_text(self, name)} ({', '.join(notes)}): {{ {d.membership_sparql} }}"
                         + (f"; {d.features}" if d.features else "") + excluded)
        return (f"Groups ({self.all_size} entities in any group; groups may overlap):\n" + "\n".join(lines))

    def with_groups(self) -> KgProfile:
        """The input profile with the groups that have entities; profiles of kept groups stay."""
        old = self.input_profile.groups
        groups = {
            name: (old[name].model_copy(update={"definition": d}) if name in old else GroupProfile(definition=d))
            for name, d in self.definitions.items() if self.sizes.get(name)
        }
        return self.input_profile.model_copy(update={"entity_space": self.space, "groups": groups}, deep=True)

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_function("group", (
                "Tests a membership pattern: how many entities of the entity space match, how much of "
                "each current group they cover, which predicates and types distinguish them from the "
                "rest of the group they overlap most (or compare_with), and examples. A pattern must "
                "be tested before it can be set.")), self.test_definition),
            ({
                "name": "set_group",
                "description": (
                    "Defines a group by a tested membership pattern, or redefines an existing one. "
                    "Groups may overlap. Shows the group sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the group."},
                        "membership": MEMBERSHIP,
                        "features": {"type": "string",
                                     "description": "What distinguishes its entities, so that they can be recognized."},
                        "parent": {"type": "string", "description": "A group it is part of, if one."},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_group),
            ({
                "name": "exclude_group",
                "description": ("Excludes a group from the alignment (entities that would not be matched for this "
                                "task), or includes it again. An entity in an excluded group is not aligned, even "
                                "if it is in other groups too."),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "reason": {"type": "string"},
                                   "excluded": {"type": "boolean", "default": True}},
                    "required": ["name", "reason"],
                    "additionalProperties": False,
                },
            }, self.exclude_group),
            ({
                "name": "remove_group",
                "description": ("Removes a group. Its entities stay in their other groups; those in no other group "
                                "are no longer aligned."),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_group),
        ]

    def set_group(self, fn_args: dict) -> str:
        name = valid_name(fn_args.get("name"), self.unit)
        membership = self.checked_membership(fn_args)
        parent = fn_args.get("parent")
        if parent is not None and self.known_name(parent) == name:
            raise FunctionCallException("A group cannot be part of itself")
        old = self.definitions.get(name)
        definition = GroupDefinition(membership_sparql=membership, parent=parent, features=fn_args.get("features") or "",
                                     excluded=old.excluded if old else False)
        self.update_definitions({**self.definitions, name: definition})
        return f"Group {name} set.\n{self.format_sizes()}"

    def excluded_share(self, definitions: dict[str, GroupDefinition]) -> float:
        excluded = [name for name, d in definitions.items() if d.excluded]
        if not excluded or not self.all_size:
            return 0.0
        grouping = Grouping(tuple(("excluded", definitions[n].membership_sparql) for n in excluded), exclusive=True)
        return group_sizes(self.manager, grouping, self.space).get("excluded", 0) / self.all_size

    def exclude_group(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        excluded = fn_args.get("excluded", True)
        definitions = {**self.definitions, name: self.definitions[name].model_copy(update={"excluded": excluded})}
        if excluded and all(d.excluded for d in definitions.values()):
            raise FunctionCallException("Not every group can be excluded: nothing would be aligned. To exclude "
                                        "part of a group, define that part as a group of its own and exclude it.")
        reason = fn_args.get("reason") or ""
        if excluded and reason:
            features = definitions[name].features
            definitions[name] = definitions[name].model_copy(
                update={"features": f"{features}; excluded: {reason}" if features else f"excluded: {reason}"})
        self.definitions = definitions
        try:
            excluded_share = self.excluded_share(definitions)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        result = f"Group {name} {'excluded' if excluded else 'included'}; {excluded_share:.0%} of the entities in groups are excluded now."
        if excluded and excluded_share > LARGE_EXCLUSION_SHARE:
            result += (" This is a large part. Exclude only entities that would not be matched for this task; "
                       "to exclude part of a group, define that part as a group of its own and exclude it.")
        return f"{result}\n{self.format_sizes()}"

    def remove_group(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last group cannot be removed")
        membership = self.definitions[name].membership_sparql
        remaining = {n: d for n, d in self.definitions.items() if n != name}
        try:
            size, overlap = membership_overlap(self.manager, group_grouping(remaining), self.space, membership)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        self.update_definitions(remaining)
        alone = overlap.get(UNCATEGORIZED, 0)
        return (f"Group {name} removed; {share(alone, size)} of its entities are in no other group and are no "
                f"longer aligned.\n{self.format_sizes()}")

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        profile = self.with_groups()
        if not profile.groups:
            raise FunctionCallException("No group has entities")
        if all(g.definition.excluded for g in profile.groups.values()):
            raise FunctionCallException("Every group is excluded: nothing would be aligned")
        self.stats = size_stats(self)
        dropped = [name for name in self.definitions if name not in profile.groups]
        summary = "Groups: " + ", ".join(
            f"{n} {self.stats['sizes'].get(n, 0)}" + (" (excluded)" if g.definition.excluded else "")
            for n, g in profile.groups.items())
        if dropped:
            summary += f"; dropped as empty: {', '.join(dropped)}"
        return profile, summary


# --- d. one group --------------------------------------------------------------------

def single_grouping(profile: KgProfile, group: str) -> Grouping:
    """The grouping of one group only: stage d looks at no other group."""
    return Grouping(((group, profile.groups[group].definition.membership_sparql),), exclusive=False)


def group_name_stats(manager: KgManager, grouping: Grouping, space: list[str], group: str,
                     roles: Roles) -> LocalNameStats:
    sample = sample_entities(manager, grouping, space, where=f'FILTER(?group = "{group}")')
    return local_name_stats(manager, sample.get(group, []), roles.names)


def group_defaults(manager: KgManager, profile: KgProfile, group: str) -> tuple[GroupProfile, LocalNameStats]:
    """Profile of a group without LLM: the local name type from its sample, its delta unchanged."""
    grouping = single_grouping(profile, group)
    current = profile.groups[group]
    usage = element_usage(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    stats = group_name_stats(manager, grouping, profile.entity_space, group, effective)
    return current.model_copy(update={"local_name_type": local_name_type(stats)}, deep=True), stats


# stage d runs in full mode for a group with at least MIN_UNASSIGNED predicates
# without a role, each covering MIN_UNASSIGNED_SHARE of its entities as subject
# (vocabulary predicates such as rdf:type not counted), or for a group formed in
# stage b2; otherwise in light mode
MIN_UNASSIGNED = 3
MIN_UNASSIGNED_SHARE = 0.05


class GroupMode(StrEnum):
    FULL = "full"    # roles of the group, findings
    LIGHT = "light"  # mainly matching pitfalls of the group as findings


def unassigned_predicates(roles: Roles, coverage: dict[str, Coverage]) -> list[tuple[str, Coverage]]:
    """Predicates of a group in no element of its roles (as predicate or
    inside a pattern), most frequent first."""
    elements = role_elements(roles)
    predicates = {key[2] for key in elements if key[1] == "predicate"}
    patterns = [element_sparql(key, value) for key, value in elements.items() if key[1] != "predicate"]
    covered = lambda p: p in predicates or any(f"<{p}>" in pattern for pattern in patterns)
    return [(p, c) for p, c in sorted(coverage.items(), key=lambda x: -x[1].entities) if not covered(p)]


def group_mode(manager: KgManager, profile: KgProfile, group: str) -> tuple[GroupMode, list[str]]:
    """(mode of stage d for a group, the unassigned predicates that count)."""
    grouping = single_grouping(profile, group)
    current = profile.groups[group]
    usage = element_usage(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    coverage = predicate_coverage(manager, grouping, profile.entity_space).get(group, {})
    size = group_sizes(manager, grouping, profile.entity_space).get(group, 0)
    counted = [p for p, c in unassigned_predicates(effective, coverage)
               if not p.startswith(VOCABULARY) and size and c.entities / size >= MIN_UNASSIGNED_SHARE]
    formed = current.definition.origin == GroupOrigin.FORMED
    return (GroupMode.FULL if formed or len(counted) >= MIN_UNASSIGNED else GroupMode.LIGHT), counted


class GroupTask(RoleEditingTask):
    name = "om-profile-group"
    stage = Stage.GROUP
    prompt_file = "profile_group.txt"
    prompt_files = {GroupMode.FULL: "profile_group.txt", GroupMode.LIGHT: "profile_group_light.txt"}
    unit = "group"

    def prepare(self, input: dict) -> str:
        profile = self.input_profile
        self.group = input.get("group")
        self.mode = GroupMode(input.get("mode") or GroupMode.FULL)
        self.prompt_file = self.prompt_files[self.mode]
        if self.group not in profile.groups:
            raise ValueError(f"Unknown group {self.group}; groups: {', '.join(profile.groups)}")
        self.space = list(profile.entity_space)
        self.grouping = single_grouping(profile, self.group)
        self.scope = [self.group]
        self.sizes = group_sizes(self.manager, self.grouping, self.space)
        self.estimates = estimated_sizes(self.manager, self.grouping, self.space)
        current = profile.groups[self.group]
        sides: dict[str, Sides] = {}
        kg_usage = element_usage(self.manager, self.grouping, self.space,
                                 roles_from_elements(role_elements(profile.roles) | role_elements(current.added)), sides)
        self.added_keys = set(role_elements(current.added))
        self.derived = derived_roles(profile.roles, kg_usage.get(self.group, {}))
        effective = effective_roles(profile.roles, current, kg_usage.get(self.group, {}))
        self.init_roles(effective, kg_usage, sides, effective)
        stats = group_name_stats(self.manager, self.grouping, self.space, self.group, effective)
        self.name_type = local_name_type(stats)
        self.stats = {"size": size_stats(self)["sizes"].get(self.group, 0), "mode": self.mode.value,
                      "local_names": stats.model_dump()}
        default = current.model_copy(update={"local_name_type": self.name_type})
        self.default = profile.model_copy(update={"groups": {**profile.groups, self.group: default}}, deep=True)

        unused = [key for key in role_elements(profile.roles) if key not in role_elements(self.derived)
                  and key not in role_elements(current.added)]
        coverage = predicate_coverage(self.manager, self.grouping, self.space).get(self.group, {})
        size = self.sizes.get(self.group, 0)
        others = [f"  {short_iri(p)}  as ?s: {share(c.entities, size)}  |  values: {format_values(c.values, c.literals, c.blanks)}"
                  for p, c in unassigned_predicates(effective, coverage)][:NUM_PREDICATES]
        definition = current.definition
        agreement = "–" if stats.agreement is None else f"{stats.agreement:.0%}"
        speaking = "–" if stats.speaking is None else f"{stats.speaking:.0%}"
        return "\n\n".join([
            f"Group {self.group}: {size_text(self, self.group)}"
            + (f"; {definition.features}" if definition.features else "")
            + f"; membership {{ {definition.membership_sparql} }}",
            f"Local names (sample of {stats.sample_size}, {stats.labelled} with a name): {agreement} agree with "
            f"a name, {speaking} look like words -> default local_name_type: {self.name_type.value}",
            "Roles of this group: the KG-wide roles used here [kg], and its own additions [group] "
            "(coverage as ?s):\n" + self.format_roles(effective, self.origin),
            "KG-wide elements not used in this group (dropped): "
            + (", ".join(short_iri(k[2]) for k in unused) or "none"),
            "Predicates of this group's entities that have no role yet: share of its entities that "
            "are their subject (as ?s), kinds of their values (?o):\n" + ("\n".join(others) or "  none"),
        ])

    def origin(self, key) -> str:
        return "group" if key in self.added_keys else "kg"

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_functions()

    def finish_properties(self) -> dict[str, dict]:
        return {"local_name_type": {
            "type": "string", "enum": [t.value for t in NameType],
            "description": "If the default is wrong: label if local names are names, code if identifiers, other if mixed.",
        }}

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.checked_roles()
        added, removed = diff_roles(self.derived, roles)
        name_type = NameType(fn_args.get("local_name_type") or self.name_type)
        current = self.default.groups[self.group]
        updated = current.model_copy(update={"local_name_type": name_type, "added": added, "removed": removed})
        profile = self.default.model_copy(update={"groups": {**self.default.groups, self.group: updated}}, deep=True)
        count = lambda r: len(role_elements(r))
        return profile, (f"Group {self.group}: local names {name_type.value}, "
                         f"{count(added)} elements added or overridden, {count(removed)} removed")


PROFILE_TASKS = [SpaceTask, PartitionsTask, RolesTask, GroupsTask, GroupTask]
