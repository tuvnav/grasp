The entities to align are reified pharmacogenomic relationships: instances of
pgxo:PharmacogenomicRelationship, each stating that patients treated with the given
drugs while having the given genetic factors may experience the given phenotypes.
They have no labels and are identified only by their components (drugs, genetic
factors such as genes and variants, phenotypes), which are linked to them by PGxO
predicates (e.g. pgxo:causes, pgxo:isAssociatedWith, pgxo:treats). Most component
individuals have no label either; use the labels of the classes they instantiate.

Domain knowledge for deciding whether a component is the same as, equivalent to or
more specific than another one:
- Predicates: pgxo:causes is more specific than pgxo:influences, which is more
  specific than pgxo:isAssociatedWith. This hierarchy is not stated in the
  ontologies themselves.
- Genetic factors: a genetic factor that is part of another one (obo:BFO_0000050,
  "part of"; e.g. a variant of a gene) is more specific than it.
- Drugs: compare the most specific ATC or ChEBI classes the drug individuals
  instantiate (rdf:type), ordered by rdfs:subClassOf.
- Phenotypes: compare the most specific MeSH classes the phenotype individuals
  instantiate (rdf:type), ordered by rdfs:subClassOf.
- Individuals linked by owl:sameAs are the same individual.
