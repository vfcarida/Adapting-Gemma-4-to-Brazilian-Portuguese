"""Unit tests for advanced merging algorithms (TIES, DARE)."""

import pytest
import torch

from src.train.residual_merge import (
    _ties_elect_and_merge,
    _trim_tensor,
    merge_tensors,
)


class TestTrimTensor:
    def test_trim_density_1(self):
        v = torch.randn(10, 10)
        trimmed = _trim_tensor(v, 1.0)
        assert torch.allclose(v, trimmed)

    def test_trim_density_0(self):
        v = torch.randn(10, 10)
        trimmed = _trim_tensor(v, 0.0)
        assert torch.allclose(trimmed, torch.zeros_like(v))

    def test_trim_partial(self):
        # 10 elements, keep top 40% (4 elements)
        v = torch.tensor([1.0, -2.0, 0.5, 0.1, -3.0, 0.0, 0.2, 4.0, -0.4, 0.3])
        # Magnitudes: [1.0, 2.0, 0.5, 0.1, 3.0, 0.0, 0.2, 4.0, 0.4, 0.3]
        # Sorted magnitudes: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0]
        # Top 4: 4.0, 3.0, 2.0, 1.0 (corresponds to values 4.0, -3.0, -2.0, 1.0)
        trimmed = _trim_tensor(v, 0.4)
        expected = torch.tensor([1.0, -2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        assert torch.allclose(trimmed, expected)


class TestTiesElectAndMerge:
    def test_basic_elect_and_merge(self):
        # Two task vectors
        v1 = torch.tensor([2.0, -1.0, 0.0])
        v2 = torch.tensor([1.0, -3.0, 0.5])
        
        # At idx 0: both positive. Consensus = +1. Agreeing = [2.0, 1.0]. Avg = 1.5
        # At idx 1: both negative. Consensus = -1. Agreeing = [-1.0, -3.0]. Avg = -2.0
        # At idx 2: v1=0 (no update), v2=0.5 (positive). Consensus = +1. Agreeing = [0.5]. Avg = 0.5
        merged = _ties_elect_and_merge([v1, v2])
        expected = torch.tensor([1.5, -2.0, 0.5])
        assert torch.allclose(merged, expected)

    def test_sign_conflict(self):
        # Conflict: v1 has +5.0, v2 has -2.0. Positive has larger magnitude (5.0 > 2.0).
        # Consensus sign = +1. Only v1 agrees (+5.0). v2 is filtered out.
        # Merged update at idx 0 is 5.0 (since only 1 update agreed with consensus).
        v1 = torch.tensor([5.0])
        v2 = torch.tensor([-2.0])
        merged = _ties_elect_and_merge([v1, v2])
        assert torch.allclose(merged, torch.tensor([5.0]))


class TestMergeTensors:
    def test_task_arithmetic(self):
        base = torch.zeros(5)
        cpt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        instruct = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        
        # W_merged = CPT + alpha * (Instruct - Base)
        # = cpt + 0.5 * instruct = [2.0, 3.0, 4.0, 5.0, 6.0]
        merged = merge_tensors(base, cpt, instruct, alpha=0.5, method="task_arithmetic", density=1.0)
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        assert torch.allclose(merged, expected)

    def test_dare_linear(self):
        base = torch.zeros(100)
        cpt = torch.ones(100)
        instruct = torch.ones(100) * 2.0
        
        # With density = 0.5, about half of the elements in the deltas should be zeroed,
        # and the rest scaled by 1/0.5 = 2.
        merged = merge_tensors(base, cpt, instruct, alpha=1.0, method="dare_linear", density=0.5, seed=42)
        
        # Verify that we have zeros
        assert (merged == 0).sum() > 0
        # Verify that scaled values are present (e.g. 1.0 * 2 = 2.0, instruct delta = 2.0 * 2 = 4.0)
        # So possible non-zero values: 2.0 (only CPT kept), 4.0 (only Instruct kept), 6.0 (both kept)
        assert torch.all((merged == 0) | (merged == 2.0) | (merged == 4.0) | (merged == 6.0))

    def test_ties(self):
        base = torch.zeros(5)
        cpt = torch.tensor([1.0, -2.0, 0.5, 3.0, 0.0])
        instruct = torch.tensor([0.2, 1.0, -3.0, 0.1, 4.0])
        
        # Trims and merges
        merged = merge_tensors(base, cpt, instruct, alpha=1.0, method="ties", density=0.6, seed=42)
        assert merged.shape == (5,)

    def test_seed_reproducibility(self):
        base = torch.zeros(100)
        cpt = torch.ones(100)
        instruct = torch.ones(100)
        
        m1 = merge_tensors(base, cpt, instruct, alpha=1.0, method="dare_linear", density=0.5, seed=42)
        m2 = merge_tensors(base, cpt, instruct, alpha=1.0, method="dare_linear", density=0.5, seed=42)
        m3 = merge_tensors(base, cpt, instruct, alpha=1.0, method="dare_linear", density=0.5, seed=43)
        
        assert torch.allclose(m1, m2)
        assert not torch.allclose(m1, m3)

    def test_preserves_dtype(self):
        base = torch.zeros(5, dtype=torch.bfloat16)
        cpt = torch.ones(5, dtype=torch.bfloat16)
        instruct = torch.ones(5, dtype=torch.bfloat16)
        
        merged = merge_tensors(base, cpt, instruct, alpha=0.5, method="dare_ties", density=0.8)
        assert merged.dtype == torch.bfloat16
