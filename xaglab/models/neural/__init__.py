"""Neural models (Phase 3 onward): shared training machinery and networks.

  data.py     feature matrix → scaled tensor, sliding windows, multi-task targets
  losses.py   the multi-task loss (BCE for direction, QLIKE for relative volatility)
  nets.py     SequenceNet (LSTM / GRU over pooled features)
  trainer.py  seeding, early-stopping split, training loop, prediction, ensembles

"""
