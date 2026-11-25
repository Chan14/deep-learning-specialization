"""Batch GD vs true SGD vs mini-batch SGD, from scratch in NumPy.

A 3-layer ReLU MLP regresses California Housing prices. The three methods
differ only in the batch iterator handed to `trainer.train`:

    data.py        load, split, and normalize California Housing
    model.py       init, forward, backward (MSE data loss only)
    losses.py      MSE data loss + L2 penalty, reported separately
    optimizer.py   one vanilla GD step, L2 added to weight gradients only
    trainer.py     batch iterators + method-agnostic training loop
    grad_check.py  centered-difference check of model.backward
"""
