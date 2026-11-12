# Third-party notices

graph-memory is released under the MIT License (see `LICENSE`). It includes or adapts
the following third-party work.

## Graphiti

- Source: https://github.com/getzep/graphiti
- Copyright 2024, Zep Software, Inc.
- License: Apache License 2.0, full text in `LICENSES/Apache-2.0.txt`

Parts of graduation and the knowledge store are adapted from Graphiti: its entity
extraction exclusions, its deterministic entity dedup (fuzzy normal form, the guard for
short or repetitive names, character 3-gram Jaccard matching, the 0.6 embedding
candidate floor), its contradiction test ("both cannot be true at once") and its
bi-temporal invalidation of edges. The code was rewritten for this project; no Graphiti
source file is included. The adapted places are marked with "Graphiti" in
`src/graph_memory/graduation.py` and `src/graph_memory/knowledge.py`.

## 3d-force-graph

- Source: https://github.com/vasturiano/3d-force-graph
- Copyright (c) 2017 Vasco Asturiano
- License: MIT, full text in `LICENSES/MIT-3d-force-graph.txt`

`src/graph_memory/ui/vendor/3d-force-graph.min.js` is the unmodified 1.80.1 UMD bundle
from npm. The bundle contains the library's own dependencies (three.js, d3-force-3d and
others), each under its own permissive license, as published with those packages.
