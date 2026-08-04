"""Torch-free tests for the V3 train/val perceptual-hash index."""

from myscripts.V3.audit_dataset_splits_v3 import HammingBKTree


def test_hamming_bk_tree_returns_only_neighbors_inside_radius():
    tree = HammingBKTree()
    tree.add(0b0000, "zero")
    tree.add(0b0001, "one")
    tree.add(0b1111, "four")
    assert tree.query(0b0011, 1) == [(1, "one")]


def test_hamming_bk_tree_preserves_duplicate_payloads():
    tree = HammingBKTree()
    tree.add(0b1010, "first")
    tree.add(0b1010, "second")
    assert tree.query(0b1010, 0) == [(0, "first"), (0, "second")]
