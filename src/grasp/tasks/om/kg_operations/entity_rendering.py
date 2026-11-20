"""
Entities shown to the LLM, the same in all stages (EntityDisplay.show), in
three views of one concept (EntityView; presets SHORT_VIEW, MIDDLE_VIEW,
LONG_VIEW):
    line     'large intestine <mouse:MA_1>', optionally with further names and
             synonyms: '... (names: "Dickdarm"@de; synonyms: "colon", +1 more)'
    context  the line, then everything the KG says about the entity (names,
             synonyms, definitions, types, class expressions, relations,
             incoming edges, usage as type or predicate), cut at a budget

What the KG means is taken from its profile (the roles: names, synonyms,
definitions, and the membership relations as types), so it works on any
schema; a profile without a role shows nothing for it (no fallback to
rdfs:label and co., which would hide gaps of the profile). Two steps:
prefetch retrieves (SPARQL, in batches, cached for the whole task), show only
formats what was retrieved.
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.om.data_models.profile import (
    INVERSE_PREFIX,
    ROLE_FIELDS,
    KgProfile,
    RoleName,
    PredicateRoles,
    StructureRelationKind,
    roles_of_entity,
)
from grasp.tasks.om.foundation.types import IRI, BindingType, BlankNodeTree
from grasp.tasks.om.foundation.rdf_vocabulary import (
    OWL_ON_PROPERTY,
    RDF_FIRST,
    RDF_REST,
    RDF_TYPE,
    VOCABULARY,
    VOCABULARY_PREFIXES,
    local_name,
)
from grasp.tasks.om.kg_operations.grouping import groups_by_entity
from grasp.tasks.om.kg_operations.candidate_generation import (
    named_value_patterns,
)
from grasp.tasks.om.sparql_access import (
    select_rows,
    ValueGroup,
    retrieve_blank_nodes,
    retrieve_value_groups,
    down_pushed_union,
    sparql_literal,
    union_of,
    as_batched_values_blocks,
    values_clause,
)

# ---------------------------------------------------------------- building blocks

# restriction predicates and their keyword in Manchester syntax
RESTRICTIONS = {
    IRI("http://www.w3.org/2002/07/owl#someValuesFrom"): "some",
    IRI("http://www.w3.org/2002/07/owl#allValuesFrom"): "only",
    IRI("http://www.w3.org/2002/07/owl#hasValue"): "value",
}

# levels of nested blank nodes resolved below an entity (class expressions);
# only a guard, render_blank_node also stops at cycles
BLANK_NODE_MAX_DEPTH = 6

# roles retrieved for every shown entity, and for the long view
STANDARD_VIEW_ROLES: tuple[RoleName, ...] = ("names", "synonyms")

# values retrieved per role or predicate of an entity: more than any view
# shows, so the views cut further and the counts stay exact
MAX_VALUES_PER_GROUP = 50


def label_with_shortened_iri(label: str | None, shortened_iri: str) -> str:
    """
    The one shape of a labelled entity: 'Label <shortened_iri>', or '<shortened_iri>'
    without a label.
    """
    return f"'{label}' <{shortened_iri}>" if label else f"<{shortened_iri}>"


def clip(text: str, max_chars: int) -> str:
    """
    Whitespace collapsed, cut at max_chars.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def shortened_vocabulary_name(iri: str) -> str | None:
    """
    owl:Class for an IRI of the standard vocabulary, else None.
    """
    for namespace, prefix in VOCABULARY_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return None


def render_blank_node(
        node: str,
        tree: BlankNodeTree,
        show_iri: Callable[[str], str],
        seen: frozenset[str] = frozenset(),
        ) -> str:
    """
    A blank node as a class expression: RDF lists as [a, b], restrictions
    as (property some filler), anything else as (predicate value; ...).
    """
    if node in seen:
        return "(…)"
    seen = seen | {node}
    by_pred: dict[IRI, list[Binding]] = {}
    for pred, obj in tree.get(node, []):
        by_pred.setdefault(pred, []).append(obj)

    def render(obj: Binding) -> str:
        if obj.typ == BindingType.BNODE:
            return render_blank_node(obj.value, tree, show_iri, seen)
        if obj.typ == BindingType.URI:
            return show_iri(obj.value)
        return f'"{obj.value}"'

    if RDF_FIRST in by_pred:
        items: list[str] = []
        visited: set[str] = set()
        current: str | None = node
        while current is not None and current not in visited:
            visited.add(current)
            current_edges = dict(tree.get(current, []))
            first = current_edges.get(RDF_FIRST)
            if first is None:
                break
            items.append(render(first))
            rest = current_edges.get(RDF_REST)
            current = rest.value if rest is not None and rest.typ == BindingType.BNODE else None
        return "[" + ", ".join(items) + "]"
    if OWL_ON_PROPERTY in by_pred:
        prop = render(by_pred[OWL_ON_PROPERTY][0])
        for pred, keyword in RESTRICTIONS.items():
            if pred in by_pred:
                return f"({prop} {keyword} {render(by_pred[pred][0])})"
    parts = [
        f"{show_iri(pred)} {', '.join(render(o) for o in objs)}"
        for pred, objs in by_pred.items()
        if pred != RDF_TYPE  # e.g. owl:Restriction, implied by the rendering
    ]
    return "(" + "; ".join(parts) + ")"


# ---------------------------------------------------------------- views

@dataclass(frozen=True)
class EntityView:
    """
    What is shown of an entity.
    Line: iri: its reference (prefixed IRI if a prefix is known); num_names:
    names shown, the first as label (0: none); num_synonyms: synonyms shown.
    Values that are not shown are counted.
    context: the line, then the context of the entity, with: num_values: values
    shown per predicate (and definitions); max_chars_per_literal; max_incoming_predicates;
    max_chars: budget of the whole context (the least important lines are cut).
    """
    iri: bool = True
    num_names: int = 1
    num_synonyms: int = 0
    context: bool = False
    num_values: int = 5
    max_chars_per_literal: int = 300
    max_incoming_predicates: int = 8
    max_chars: int = 2000


SHORT_VIEW = EntityView()                                # Label <mouse:MA_1>
LONG_VIEW = EntityView(num_names=3, num_synonyms=10, context=True)


def name_order(value: Binding) -> tuple[str, str]:
    """
    Order of names, synonyms and definitions: without language first, then
    by language and text (so that the label is the same in every run).
    """
    return (value.lang or "", value.value)


def render_as_literal(
        value: Binding, max_chars: int | None = None
        ) -> str:
    text = clip(value.value, max_chars) if max_chars is not None else value.value
    return f'"{text}"' + (f"@{value.lang}" if value.lang else "")


def shown_values(
        title: str,
        values: list[Binding],
        num_to_show: int,
        max_chars: int | None = None
        ) -> str | None:
    """
    'title: "a", "b", +2 more', None if nothing is to be shown.
    """
    if num_to_show <= 0 or not values:
        return None
    shown_values = ", ".join(render_as_literal(value, max_chars)
                             for value in values[:num_to_show])
    cutoff_values = f", +{len(values) - num_to_show} more" if len(values) > num_to_show else ""
    return f"{title}: {shown_values}{cutoff_values}"


# ---------------------------------------------------------------- what the profile says

def plain_predicates(
        roles: PredicateRoles, role_names: Iterable[RoleName]
        ) -> set[IRI]:
    """
    The plain predicates of the roles (not their SPARQL patterns).
    """
    patterns = [
        roles.predicate_pattern_by_name(role_name) for role_name in role_names
        ]
    return {
        predicate for pattern in patterns if pattern is not None
        for predicate in pattern.predicates
        }


def membership_predicates(roles: PredicateRoles) -> list[IRI]:
    """
    The plain predicates of the membership relations: the types of an entity.
    """
    return sorted({
        predicate for relation in roles.structure
        if relation.kind == StructureRelationKind.MEMBERSHIP
        for predicate in relation.predicate_pattern.predicates
    })


def context_role_names(roles: PredicateRoles) -> list[str]:
    """
    The roles shown in a context: all but noise, the structure relations in
    both directions.
    """
    structural_relation_names = [relation.name for relation in roles.structure]
    return [
        *(name for name in ROLE_FIELDS if name != "noise"),
        *structural_relation_names,
        *(INVERSE_PREFIX + name for name in structural_relation_names),
    ]


def role_branches(roles: PredicateRoles, role_names: Iterable[str]) -> list[str]:
    """
    SPARQL alternatives over ?s (entity), ?predicate_ref (role name) and ?o (value),
    for retrieve_value_groups; roles that are not set are left out.
    """
    return [
        f"VALUES ?predicate_ref {{ {sparql_literal(name)} }}\n{branch}"
        for name, pattern in named_value_patterns(list(role_names), roles)
        for branch in pattern.branches_for_union()
    ]


# ------------------------------------------------- context: extra retrieved data

@dataclass
class PredicateUsage:
    """
    Entity is used as a predicate/property
    """
    triple_count: int
    subject_types: list[tuple[IRI, int]]  # most common subject types with count
    object_types: list[tuple[IRI, int]]  # most common object types with count
    num_literal_objects: int
    examples: list[tuple[Binding, Binding]]  # subject - object pairs


@dataclass
class EntityContext:
    """
    What the KG says about an entity besides its role values (those are in
    EntityDisplay.values_by_entity_role): the edges of all other predicates.
    """
    iri: IRI
    outgoing: dict[str, ValueGroup]  # predicate -> objects
    incoming: dict[str, ValueGroup]  # predicate -> subjects
    used_as_predicate: PredicateUsage | None
    blank_nodes: BlankNodeTree = field(default_factory=dict)
    # IRIs that only carry a value (e.g. synonym nodes): their name stands for them
    helper_nodes: set[IRI] = field(default_factory=set)


# ---------------------------------------------------------------- context: gathering

def find_helper_nodes(
        manager: KgManager, iris: Iterable[IRI], roles: PredicateRoles
        ) -> set[IRI]:
    """
    The IRIs that only carry a value: at least one name, and otherwise only
    edges of plain predicates of the names, synonyms and membership roles (e.g.
    definition and synonym nodes in the OAEI anatomy ontologies). Their name
    stands for them like a literal. Patterns of the profile cannot be checked
    here, only plain predicates.
    """
    names = roles.predicate_pattern_by_name("names")
    if names is None:
        return set()
    plain_preds = sorted(
        plain_predicates(roles, STANDARD_VIEW_ROLES)
        | set(membership_predicates(roles))
        )
    named: set[IRI] = set()
    with_other_edges: set[IRI] = set()
    for values in as_batched_values_blocks(iris):
        has_name_sparql = down_pushed_union(
            f"VALUES ?s {{ {values} }}", names.branches_for_union()
            )
        named |= {
            IRI(row["s"].value)
            for row in select_rows(
                manager, f"SELECT DISTINCT ?s WHERE {{\n{has_name_sparql}\n}}"
                )
            }
        not_plain = (
            f"FILTER(?p NOT IN ({', '.join(f'<{p}>' for p in plain_preds)}))"
            if plain_preds else ""
        )
        with_other_edges |= {
            IRI(row["s"].value)
            for row in select_rows(
                manager, f"SELECT DISTINCT ?s WHERE {{ VALUES ?s {{ {values} }} ?s ?p ?o {not_plain} }}"
                )
            }
    return named - with_other_edges


def retrieve_predicate_usage(
        manager: KgManager, entity: IRI, type_predicates: list[IRI]
        ) -> PredicateUsage | None:
    term = prepare_identifier_for_sparql(entity, manager.iri_literal_parser)
    rows = select_rows(manager, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?usage_subject {term} ?usage_object }}")
    count = int(rows[0]["n"].value) if rows else 0
    if not count:
        return None

    def top_types(position: str) -> list[tuple[IRI, int]]:
        if not type_predicates:
            return []
        return [
            (IRI(r["t"].value), int(r["n"].value)) for r in select_rows(
                manager,
                f"SELECT ?t (COUNT(*) AS ?n) WHERE {{ ?usage_subject {term} ?usage_object . ?{position} {union_of(type_predicates)} ?t }} "
                f"GROUP BY ?t ORDER BY DESC(?n) LIMIT 3",
            )
        ]

    literals = select_rows(
        manager, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?usage_subject {term} ?usage_object FILTER(isLiteral(?usage_object)) }}"
        )
    examples = select_rows(
        manager, f"SELECT ?usage_subject ?usage_object WHERE {{ ?usage_subject {term} ?usage_object }} LIMIT 3"
        )
    return PredicateUsage(
        triple_count=count,
        subject_types=top_types("usage_subject"),
        object_types=top_types("usage_object"),
        num_literal_objects=int(literals[0]["n"].value) if literals else 0,
        examples=[(r["usage_subject"], r["usage_object"]) for r in examples],
    )


# ---------------------------------------------------------------- the display

@dataclass
class EntityDisplay:
    """
    How the entities of one KG are shown. Each entity with the roles of its
    groups (entity_roles); objects outside the entity space without a name
    are only counted in a context (e.g. images).
    """
    manager: KgManager
    kg_profile: KgProfile
    # entity -> role name (with INVERSE_PREFIX: the other direction) -> values;
    # retrieved entities without any value have an empty entry
    values_by_entity_role: dict[IRI, dict[str, ValueGroup]] = field(default_factory=dict)
    # entity -> its context, for the long view
    contexts_by_entity: dict[IRI, EntityContext] = field(default_factory=dict)

    # ------------------------------------------------------------ retrieval

    def prefetch(self, entities: Iterable[IRI]) -> None:
        """
        Retrieves all relevant information for the specified entities:
        the values of all their roles, the edges of all other predicates
        and the names of all IRIs in them.
        """
        entities = set(entities)
        # only prefetch  which are not in cache
        new_entities = entities - self.contexts_by_entity.keys()
        for roles, members in self.roles_by_group_names(new_entities).values():
            # main entities: retrieve all roles (also if retrieved before as a neighbour with names only)
            self.prefetch_role_values(members, context_role_names(roles), roles)
            # the plain predicates of all roles (also noise) should be removed from 'other edges'
            role_predicates = plain_predicates(
                roles, [*ROLE_FIELDS, *(relation.name for relation in roles.structure)]
            )
            removed_role_predicate = (
                f"FILTER(?predicate_ref NOT IN ({', '.join(f'<{p}>' for p in sorted(role_predicates))}))"
                if role_predicates else ""
            )
            noise_filter = "".join(
                f" FILTER NOT EXISTS {{{values_clause('s', members)} {branch}}}"
                for branch in roles.noise.patterns
                ) if roles.noise is not None else ""
            outgoing = retrieve_value_groups(
                self.manager,
                members,
                [f"?s ?predicate_ref ?o {removed_role_predicate}{noise_filter}"],
                MAX_VALUES_PER_GROUP
            )
            incoming = retrieve_value_groups(
                self.manager,
                members,
                [f"?o ?predicate_ref ?s FILTER(isIRI(?o)) {removed_role_predicate}"],
                MAX_VALUES_PER_GROUP,
            )
            for entity in sorted(members):
                self.contexts_by_entity[entity] = self.build_context(
                    entity, outgoing.get(entity, {}), incoming.get(entity, {}), roles
                )
        # neighbours: names only, if not retrieved yet
        neighbours = {iri for entity in entities for iri in self.shown_iris(entity)}
        for roles, members in self.roles_by_group_names(neighbours - self.values_by_entity_role.keys()).values():
            self.prefetch_role_values(members, STANDARD_VIEW_ROLES, roles)

    def roles_by_group_names(self, entities: set[IRI]) -> dict[frozenset[str], tuple[PredicateRoles, set[IRI]]]:
        """
        The entities divided by their combination of groups, each part with
        its roles: entities in the same groups have the same roles, so each
        part can be retrieved in one batch.
        """
        divided: dict[frozenset[str], set[IRI]] = {}
        for entity, group_names in groups_by_entity(
                self.manager, self.kg_profile, entities
                ).items():
            divided.setdefault(group_names, set()).add(entity)
        return {
            group_names: (
                roles_of_entity(self.kg_profile.roles,
                             map(self.kg_profile.get_group, group_names)),
                members,
            )
            for group_names, members in divided.items()
        }

    def prefetch_role_values(self, entities: set[IRI], role_names: Iterable[str], roles: PredicateRoles) -> None:
        """
        The values of the specified roles. Vocabulary IRIs are skipped.
        """
        entities = {entity for entity in entities if not entity.startswith(VOCABULARY)}
        value_groups = retrieve_value_groups(
            self.manager, entities, role_branches(
                roles, role_names
                ), MAX_VALUES_PER_GROUP
        )
        for entity in entities:
            self.values_by_entity_role[entity] = value_groups.get(entity, {})

    def build_context(
            self,
            entity: IRI,
            outgoing: dict[str, ValueGroup],
            incoming: dict[str, ValueGroup],
            roles: PredicateRoles,
            ) -> EntityContext:
        """
        The context of a main entity from its retrieved edges: blank nodes
        resolved, helper nodes found, usage as predicate.
        """
        groups = [
            *self.values_by_entity_role.get(entity, {}).values(),
            *outgoing.values(), *incoming.values()
            ]
        has_blank_nodes = any(
            value.typ == BindingType.BNODE for group in groups for value in group.values
        )
        # helper node candidates: IRI objects of the other edges (role values are entities)
        candidates = {
            IRI(value.value) for group in outgoing.values() for value in group.values
            if value.typ == BindingType.URI
        }
        return EntityContext(
            iri=entity,
            outgoing=outgoing,
            incoming=incoming,
            used_as_predicate=retrieve_predicate_usage(self.manager, entity, membership_predicates(roles)),
            blank_nodes=retrieve_blank_nodes(self.manager, entity, BLANK_NODE_MAX_DEPTH) if has_blank_nodes else {},
            helper_nodes=find_helper_nodes(self.manager, candidates, roles),
        )

    def shown_iris(self, entity: IRI) -> set[IRI]:
        """All IRIs of the context of a main entity that may be shown with their name."""
        context = self.contexts_by_entity[entity]
        groups = [
            *self.values_by_entity_role.get(entity, {}).values(),
            *context.outgoing.values(),
            *context.incoming.values(),
        ]
        iris = {
            IRI(value.value) for group in groups for value in group.values
            if value.typ == BindingType.URI
        }
        for edges in context.blank_nodes.values():
            for pred, obj in edges:
                iris.add(pred)
                if obj.typ == BindingType.URI:
                    iris.add(IRI(obj.value))
        if context.used_as_predicate:
            usage = context.used_as_predicate
            iris.update(t for t, _ in usage.subject_types + usage.object_types)
            for s, o in usage.examples:
                iris.update(IRI(b.value) for b in (s, o) if b.typ == BindingType.URI)
        return iris

    # ------------------------------------------------------------ showing

    def values(self, entity: IRI, role_name: RoleName) -> list[Binding]:
        """
        The literal values of a role in name order (IRIs and blank nodes
        are shown in a context only).
        """
        group = self.values_by_entity_role.get(entity, {}).get(role_name)
        if group is None:
            return []
        return sorted((value for value in group.values if value.typ == BindingType.LITERAL), key=name_order)

    def has_name(self, iri: str) -> bool:
        return bool(self.values(IRI(iri), "names"))

    def show(self, entity: IRI, view: EntityView = SHORT_VIEW) -> str:
        """
        The entity by the view (None: the display's own view): a line, or
        the line with the context (if prefetched with a context view).
        """
        if view.context and entity in self.contexts_by_entity:
            return self.render_context(self.contexts_by_entity[entity], view)
        return self.render_line(entity, view)

    def render_line(self, entity: str, view: EntityView = SHORT_VIEW) -> str:
        """
        e.g. 'large intestine <mouse:MA_1> 
        (names: "Dickdarm"@de; synonyms: "colon", +1 more)';
        a vocabulary IRI as owl:Class."""
        vocabulary_name = shortened_vocabulary_name(entity)
        if vocabulary_name is not None:
            return vocabulary_name
        names = self.values(IRI(entity), "names")
        label = names[0].value if names and view.num_names > 0 else None
        reference = self.manager.format_iri(
            entity
            ) if view.iri or label is None else None
        text = label_with_shortened_iri(
            label, reference
            ) if reference is not None else str(label)
        details = [
            detail for detail in (
                shown_values("names", names[1:], view.num_names - 1),
                shown_values("synonyms", self.values(IRI(entity), "synonyms"), view.num_synonyms),
            ) if detail
        ]
        return text + (f" ({'; '.join(details)})" if details else "")

    def render_value(self, value: Binding, context: EntityContext, view: EntityView) -> str:
        """
        A value in a context: a class expression, the name of a helper
        node, the line of an entity or a literal.
        """
        if value.typ == BindingType.BNODE:
            return render_blank_node(value.value, context.blank_nodes, self.render_line)
        if value.typ == BindingType.URI:
            names = self.values(IRI(value.value), "names")
            if IRI(value.value) in context.helper_nodes and names:
                return render_as_literal(names[0], view.max_chars_per_literal)
            return self.render_line(value.value)
        return render_as_literal(value, view.max_chars_per_literal)

    def render_value_group(
            self, title: str,
            value_group: ValueGroup,
            context: EntityContext,
            view: EntityView
            ) -> str:
        """
        'title (count): a, b [+n more; m unnamed values outside the entity
        space]'; named values first. IRIs outside the entity space without a
        name (e.g. images) are only counted.
        """
        def is_unnamed_outside_space(value: Binding) -> bool:
            return (
                value.typ == BindingType.URI and bool(self.kg_profile.entity_space)
                and not value.value.startswith(tuple(self.kg_profile.entity_space))
                and not value.value.startswith(VOCABULARY) and not self.has_name(value.value)
            )

        shown = [
            value for value in value_group.values if not is_unnamed_outside_space(value)
            ]
        hidden = len(value_group.values) - len(shown)
        shown.sort(
            key=lambda value: value.typ == BindingType.URI
            and not self.has_name(value.value)
            and not value.value.startswith(VOCABULARY)
            )
        notes = []
        not_shown = value_group.total_count - min(len(shown), view.num_values) - hidden
        if not_shown > 0:
            notes.append(f"+{not_shown} more")
        if hidden:
            notes.append(f"{hidden} unnamed values outside the entity space")
        values = ", ".join(
            self.render_value(value, context, view) for value in shown[:view.num_values]
            )
        return f"{title} ({value_group.total_count}): {values}" + (
            f" [{'; '.join(notes)}]" if notes else ""
            )

    # TODO: return Section or Item, must only change last
    # line for this. Render with format_display_content
    # in callers or directly here.
    def render_context(self, context: EntityContext, view: EntityView) -> str:
        """
        The line of the entity, then its context in a fixed order of
        importance, cut at view.max_chars. Pure: no SPARQL.
        """
        lines: list[str] = []

        # names (besides the label in the header), synonyms (not equal to a
        # name), definitions: from the roles of the profile
        names = self.values(context.iri, "names")
        name_texts = {name.value for name in names}
        synonyms = [
            synonym for synonym in self.values(context.iri, "synonyms")
            if synonym.value not in name_texts
            ]
        definitions = self.values(context.iri, "definitions")
        for line in (
            shown_values("further names", names[1:], len(names)),
            shown_values("synonyms", synonyms, view.num_synonyms),
            shown_values(
                "definitions",
                definitions,
                view.num_values,
                view.max_chars_per_literal
                ),
        ):
            if line:
                lines.append(line)

        # the other roles, structure relations in both directions
        for role_name, group in self.values_by_entity_role.get(context.iri, {}).items():
            if role_name not in ("names", "synonyms", "definitions"):
                lines.append(self.render_value_group(role_name, group, context, view))

        # the edges of all other predicates: predicates with few values first
        for direction, groups in (("", context.outgoing), ("incoming ", context.incoming)):
            for predicate, group in sorted(groups.items(), key=lambda item: item[1].total_count):
                lines.append(
                    self.render_value_group(
                        direction + local_name(predicate), group, context, view
                        )
                    )

        # usage by other entities
        if context.used_as_predicate:
            usage = context.used_as_predicate

            def type_counts(types: list[tuple[IRI, int]]) -> str:
                return ", ".join(f"{self.render_line(t)} {n}" for t, n in types) or "untyped"

            def example(binding: Binding) -> str:
                if binding.typ == BindingType.URI:
                    return self.render_line(binding.value)
                if binding.typ == BindingType.BNODE:
                    return "_"
                return f'"{clip(binding.value, view.max_chars_per_literal)[:60]}"'

            examples = "; ".join(f"{example(s)} -> {example(o)}" for s, o in usage.examples)
            lines.append(
                f"used as predicate in {usage.triple_count} triples; subject types: "
                f"{type_counts(usage.subject_types)}; object types: {type_counts(usage.object_types)}; "
                f"{usage.num_literal_objects} literal objects; e.g. {examples}"
            )

        # budget: sections are ordered by importance, so the least important are cut
        header = "Entity " + self.render_line(context.iri, EntityView(num_names=1))
        output = [header]
        used_chars = len(header)
        for index, line in enumerate(lines):
            if used_chars + len(line) + 3 > view.max_chars:
                output.append(f"  … ({len(lines) - index} more lines omitted)")
                break
            output.append(f"  {line}")
            used_chars += len(line) + 3
        return "\n".join(output)
