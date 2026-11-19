"""
Layer 2: generic SPARQL access (execution, VALUES, query building, lookups), without domain logic.

Variable convention of all queries and patterns: ?s is the entity a query is
about (also where the entity is the object of a triple, e.g. incoming edges
?o ?p ?s), ?o its value, ?p the predicate between them. Any further entity gets
a speaking name (e.g. ?linked, ?usage_subject).
"""
