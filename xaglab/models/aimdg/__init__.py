"""AIM-DG: the Adaptive Inter-Market Driver Graph.

The static GNN's network with a monthly choice of active edges, transferred from PG-GFE
(Movahed, Toosi & Nael, 2026):

  reliability.py   edge reliability scores: consistency − β·uncertainty of each edge's
                   predictive contribution over 21-day sub-windows (PG-GFE Eqs. 2–3)  [4.2]
  evolution.py     mask scoring by forward pass (ensemble, cached), fitness (Eq. 7), the
                   market-time clock (Eqs. 6, 11), the DEAP search with warm start and
                   elitism, and the exhaustive check over all 2^15 masks              [4.4]
  finalists.py     fine-tuning finalists (D-14 stage 2): tested and dropped (D-36)      [4.5]
  forecaster.py    AimDGLearner (every fit ends with edge selection) and AimDG, the
                   stateful monthly model with its attribution read-out, registered
                   as `aimdg`                                                        [4.6]
The edge-dropout supernet itself lives in neural/nets.py (StaticGraphNet(edge_dropout=…)) [4.3].
"""
