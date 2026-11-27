## Unbedingt notwendig
- Berechenbare Relationen berechnen statt setzen: Subsumption ↔ Membership: Aus den Partitionen ableitbar: ?s Instanz und ?o Klasse → Membership; beide in derselben Partition → Subsumption
- Ablation: DEFAULTS ein- und ausschaltbar
- '### Example entities without an anchor' in der Ankerstufe keine klare Aussage, warum keine Anker trotz Kandidaten
- doppelte '## Knowledge graphs'-ÜBerschrift in Ankerzertifizierung (1-Shot)
- Kein guter Übergang Knowledge Graphs und 'Consider the following entities:' in Ankerzertifizierung
- Examples bei Ensemble Evaluation (candidates) sollten eine eigene Subsection sein... Vielleicht stärkere Strata zuerst

## Optimierungsideen
- "Default canon checked: ..." in Profiling.roles: Als Liste statt Textblock 
- partition_names_by_entity zentralisieren (nur einmal abrufen?)
- 'other'-Rollen irgendwie zwischen KGs ablgeichen.
- Rules oder in anderen Promptteil in CandidateTask: Ergänzende Strategien suchen mit unterschiedlichen Fehlerkorrelationen
- CandidateTask für große KGs pro Gruppe?
- run-0 und compare_to in anchors wieder aufnehmen
- Infos zwischen Kandidatstufen und vergangenen Stufen: Welche Strategien wurden bereits genutzt (mit Ergebnissen - z. B. wie viel Prozent Abnahme)
- Funktion für Judges: EntityContext für beliebige Entität aufrufen
- Funktion für Judges: in bestehende Äquivalenzen reinschauen (z. B. nach Partitionspaaren / Gruppenpaaren / random)
- Erinnerung, dass LLM mehrere Tools pro Runde nutzen kann, wenn es viele Runden lang immer nur einen Tool Call tätigt?
- ProfileSpaceTask: In YAML neben space auch angeben: veränderbar oder nicht.
- max_steps mit KG-Größen anpassen

## Optimierungsideen aus Probeläufen
### Profiling:
- Roles: 
    - Präfixierte Prädikate in PredicatePattern korrekt behandeln!
- Strukturrollen, die über Blank Nodes gehen, zusätzlich ein Beispielpaar zeigen, das das Modell bestätigen muss / Strukturrollen möglichst Default + berechnen
- Shortcut für Profilierung bei standartisierten KGs:
    - Space: Präfix bis einschließlich # bzw. letztem / der deklarierten Entitäten statt nur des Hosts. owl:Ontology-Knoten aus dem Space ausschließen.
    - Partitionen: class/instance konkurrieren nicht, wenn kein Punning vorliegt.
    - Namen und Annotationen: Synonyme über einen Zwischenknoten: oboInOwl:has*Synonym → Knoten → rdfs:label. Weitere Synonym-Prädikate: oboInOwl:hasBroadSynonym, oboInOwl:hasNarrowSynonym (getrennt von den exakten Synonymen), skos:hiddenLabel. Weitere Namens-Prädikate: dc:title, dcterms:title, foaf:name, schema:name. Eigene Annotationsproperties mit rdfs:subPropertyOf eines bekannten Prädikats in dessen Rolle übernehmen.
    - Struktur: subclass_of: Union-Glieder ⊑ Klasse aus owl:equivalentClass/owl:unionOf. subclass_of: Klasse ⊑ benannte Konjunkte aus owl:equivalentClass/owl:intersectionOf und aus rdfs:subClassOf/owl:intersectionOf. same_as: nur IRI-Objekte (FILTER(isIRI(?o))). inverse_of: owl:inverseOf, symmetrisch. domain: rdfs:domain, IRI-Objekte und Union-Glieder. range: rdfs:range, IRI-Objekte und Union-Glieder, ohne xsd-Typen. disjoint_with ergänzen um:owl:AllDisjointClasses (paarweise): owl:disjointUnionOf (Glieder paarweise); owl:complementOf in owl:equivalentClass/owl:intersectionOf. existential_restriction: auch unter owl:equivalentClass und innerhalb von owl:intersectionOf. restriction_on_property: Klasse → Property aus beliebigen Restriktionen (onProperty, auch Kardinalitäten).
    - Abdeckungsprüfung (Schnellweg): Prüfen, ob jede Entität im Space eine OWL-Art hat und kein Punning vorliegt. Rest = Prädikate an Entitäten im Space, die keine Default-Rolle abdeckt. Ist er leer, entfällt die LLM-Profilierung; sonst bekommt das LLM nur den Rest.

- Kandidatenstrategie: TF-IDF: Stopwortentfernung mit Sprachangabe als Option
- Kandidatenstrategie: deterministische Zusammenführung von Familien
- Präfixe in describe_profile kürzen: Auf KgManager zugreifen (Setup muss davor gelaufen sein) oder eigene Funktionalität für gekürzte Präfixe umsetzen...


## Komische Stellen in Prompts:
- Profiling common: "inverse_of is only for two relations that cannot be expressed as one"; " Before you state a number that compares entities (e.g. how many names collide), exclude the entity itself and helper nodes (e.g. nodes that hold the synonyms or values of the entity) from the comparison, and check the count with a second query."