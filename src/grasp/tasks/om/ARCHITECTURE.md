# Architecture of `grasp.tasks.om`

The ontology matching code is split into five layers. Each layer is a package;
within a layer, each topic has its own module.

**Import rule:** a module imports only from its own layer or from lower layers,
never from a higher one. `Scripts/om_orchestrator/GRASP_tests/test_layer_architecture.py`
checks this for every module.

| Layer | Package | Contents | Must not |
|---|---|---|---|
| 1 | `foundation/` | basic types (`IRI`, `Side`), vocabulary and predicate canon, kind memberships, configuration access, text normalization | access a KG |
| 2 | `sparql_access/` | generic SPARQL access: running queries, VALUES blocks, building query text, single-entity lookups | contain domain logic (profiles, groups, anchors) |
| 3 | `data_models/` | pydantic models and dataclasses: profile, scope, candidate generators, alignment state, anchors, verification, relations, correspondences, value rows | do I/O |
| 4 | `kg_operations/` | domain queries and computations on KGs: groupings, profile statistics, large-KG sampling, value retrieval, key comparison, entity context and rendering | contain GRASP tasks or prompts |
| 5 | `tasks/` | GRASP tasks with their prompts (`tasks/prompts/`) and LLM functions; certification runs judge tasks | — |

## Where does a new piece of code go?

- A type, constant or IRI helper used everywhere, without KG access → `foundation/`.
- A helper that only builds or runs SPARQL, whatever the query means → `sparql_access/`.
- A data structure that is stored, passed between stages or validated → `data_models/`.
- A function that asks a KG something with domain meaning (members of a group,
  coverage of a predicate, values of entities) or computes on such results → `kg_operations/`.
- Everything the LLM sees or calls (task setup, functions, prompts) → `tasks/`.

If a lower layer seems to need something from a higher one, the shared part
belongs lower (e.g. the names of the normalization levels are in
`foundation/text_normalization.py`, so that `data_models/scope.py` can validate
them).

## Main flows

- **KG profile** (`tasks/profile_tasks.py`, stages a, b1, c, b2, d): statistics from
  `kg_operations/profile_statistics.py` over groupings (`kg_operations/grouping.py`),
  result `data_models/profile.py`.
- **Anchor stage** (`tasks/anchor_task.py`): key rows from
  `kg_operations/anchor_key_retrieval.py`, anchor rule in `kg_operations/key_comparison.py`,
  models in `data_models/anchors.py`. To be migrated to groups and `ValueRow`.
- **Candidate stage** (`tasks/candidate_task.py`, in progress): generators from
  `data_models/candidate_generators.py`, values from `kg_operations/value_retrieval.py`
  as `data_models/value_rows.py`, competition from `data_models/alignment_state.py`.
- **Verification and certification** (`tasks/verify_task.py`, `tasks/certification.py`):
  entity context from `kg_operations/entity_context.py`, models in `data_models/verification.py`.
