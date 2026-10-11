The entities to align are reified pharmacogenomic relationships: instances of
pgxo:PharmacogenomicRelationship, each stating that patients treated with the given
drugs while having the given genetic factors may experience the given phenotypes.
They have no labels and are identified only by their components (drugs, genetic
factors such as genes and variants, phenotypes). Most component individuals have no
label either; use the labels of the classes they instantiate.

Domain knowledge for deciding whether a component is the same as, equivalent to or
more specific than another one:
- Predicates linking components to tuples, from the most general to the most
  specific (PGxO ontology; these axioms are not part of the ontologies to align):
  - pgxo:isAssociatedWith (symmetric) is the most general predicate.
  - pgxo:influences, pgxo:isInfluencedBy, pgxo:treats and pgxo:isTreatedBy are
    more specific than pgxo:isAssociatedWith.
  - pgxo:causes, pgxo:increases, pgxo:decreases, pgxo:increasesChanceOf and
    pgxo:decreasesChanceOf are more specific than pgxo:influences.
  - pgxo:isCausedBy, pgxo:isIncreasedBy, pgxo:isDecreasedBy,
    pgxo:hasChanceIncreasedBy and pgxo:hasChanceDecreasedBy are more specific than
    pgxo:isInfluencedBy.
  - Inverse pairs: pgxo:influences / pgxo:isInfluencedBy, pgxo:treats /
    pgxo:isTreatedBy, pgxo:causes / pgxo:isCausedBy, pgxo:increases /
    pgxo:isIncreasedBy, pgxo:decreases / pgxo:isDecreasedBy,
    pgxo:increasesChanceOf / pgxo:hasChanceIncreasedBy, pgxo:decreasesChanceOf /
    pgxo:hasChanceDecreasedBy.
- Genetic factors: a genetic factor that is part of another one (obo:BFO_0000050
  "part of", inverse obo:BFO_0000051 "has part"; e.g. a variant of a gene) is more
  specific than it.
- Drugs: compare the most specific ATC or ChEBI classes the drug individuals
  instantiate (rdf:type), ordered by rdfs:subClassOf.
- Phenotypes: compare the most specific MeSH classes the phenotype individuals
  instantiate (rdf:type), ordered by rdfs:subClassOf.
- Individuals linked by owl:sameAs are the same individual.
- obo:RO_0002502 ("depends on") links complex phenotypes to the phenotypes and
  drugs they refer to (e.g. warfarin-induced bleeding depends on warfarin and
  bleeding). It is not used to compare components.
