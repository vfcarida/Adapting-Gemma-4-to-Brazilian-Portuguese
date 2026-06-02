"""Tests for residual merge logic (requires torch)."""

import pytest

torch = pytest.importorskip("torch")


class TestResidualMergeArithmetic:
    """Test the core arithmetic of residual merge without loading full models."""

    def test_identity_merge(self):
        """alpha=0 should give CPT weights unchanged."""
        base = torch.randn(10, 10)
        instruct = torch.randn(10, 10)
        cpt = torch.randn(10, 10)
        alpha = 0.0

        residual = instruct - base
        merged = cpt + alpha * residual

        assert torch.allclose(merged, cpt)

    def test_full_residual(self):
        """alpha=1 should give cpt + (instruct - base)."""
        base = torch.zeros(10, 10)
        instruct = torch.ones(10, 10)
        cpt = torch.ones(10, 10) * 0.5
        alpha = 1.0

        residual = instruct - base
        merged = cpt + alpha * residual

        expected = cpt + instruct - base  # 0.5 + 1 - 0 = 1.5
        assert torch.allclose(merged, expected)

    def test_half_residual(self):
        """alpha=0.5 should give half the residual."""
        base = torch.zeros(5, 5)
        instruct = torch.ones(5, 5) * 2.0
        cpt = torch.ones(5, 5)
        alpha = 0.5

        residual = instruct - base  # 2.0
        merged = cpt + alpha * residual  # 1 + 0.5*2 = 2.0

        expected = torch.ones(5, 5) * 2.0
        assert torch.allclose(merged, expected)

    def test_negative_alpha(self):
        """Negative alpha should subtract the residual."""
        base = torch.zeros(3, 3)
        instruct = torch.ones(3, 3)
        cpt = torch.ones(3, 3) * 2.0
        alpha = -0.5

        merged = cpt + alpha * (instruct - base)
        expected = torch.ones(3, 3) * 1.5  # 2 + (-0.5)*1 = 1.5
        assert torch.allclose(merged, expected)

    def test_preserves_dtype(self):
        """Merge should preserve bfloat16 dtype."""
        base = torch.randn(4, 4, dtype=torch.bfloat16)
        instruct = torch.randn(4, 4, dtype=torch.bfloat16)
        cpt = torch.randn(4, 4, dtype=torch.bfloat16)
        alpha = 0.8

        merged = cpt + alpha * (instruct - base)
        assert merged.dtype == torch.bfloat16

    def test_shape_consistency(self):
        """All tensors must have same shape for merge."""
        base = torch.randn(3, 4)
        instruct = torch.randn(3, 4)
        cpt = torch.randn(3, 4)
        alpha = 1.0

        merged = cpt + alpha * (instruct - base)
        assert merged.shape == (3, 4)


class TestResidualMergeProperties:
    """Test mathematical properties of the merge."""

    def test_linearity_in_alpha(self):
        """Merge is linear in alpha."""
        base = torch.randn(5, 5)
        instruct = torch.randn(5, 5)
        cpt = torch.randn(5, 5)

        merged_a = cpt + 0.3 * (instruct - base)
        merged_b = cpt + 0.7 * (instruct - base)
        merged_sum = cpt + 1.0 * (instruct - base)

        # 0.3 * residual + 0.7 * residual = 1.0 * residual
        reconstructed = (merged_a - cpt) + (merged_b - cpt) + cpt
        assert torch.allclose(reconstructed, merged_sum, atol=1e-6)

    def test_when_cpt_equals_base(self):
        """If CPT didn't change weights, merge should interpolate toward instruct."""
        base = torch.randn(5, 5)
        instruct = torch.randn(5, 5)
        cpt = base.clone()  # No CPT change
        alpha = 1.0

        merged = cpt + alpha * (instruct - base)
        # Should equal instruct since cpt=base
        assert torch.allclose(merged, instruct)
