"""Tests for standalone loss helpers."""

import math
import pytest
import torch

from genkit.loss import barron_loss


def test_barron_loss_matches_half_squared_error_at_alpha_two():
    pred = torch.tensor([1.0, 3.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)

    loss = barron_loss(pred, target, alpha=2.0, scale=1.0, reduction="none")

    assert torch.allclose(loss, 0.5 * (pred - target).square())


def test_barron_loss_matches_log_case_at_alpha_zero():
    pred = torch.tensor([1.0, 3.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)

    loss = barron_loss(pred, target, alpha=0.0, scale=2.0, reduction="none")
    expected = torch.log1p(0.5 * ((pred - target) / 2.0).square())

    assert torch.allclose(loss, expected)


def test_barron_loss_matches_negative_infinity_case():
    pred = torch.tensor([1.0, 3.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)

    loss = barron_loss(pred, target, alpha=-math.inf, scale=1.5, reduction="none")
    expected = -torch.expm1(-0.5 * ((pred - target) / 1.5).square())

    assert torch.allclose(loss, expected)


def test_barron_loss_supports_tensor_alpha_and_sum_reduction():
    pred = torch.tensor([1.0, 2.0], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0], dtype=torch.float64)
    alpha = torch.tensor([2.0, 0.0], dtype=torch.float64)

    loss = barron_loss(pred, target, alpha=alpha, scale=1.0, reduction="sum")
    expected = 0.5 * pred[0].square() + torch.log1p(0.5 * pred[1].square())

    assert loss == pytest.approx(expected.item())


def test_barron_loss_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="Invalid reduction"):
        barron_loss(torch.ones(2), torch.zeros(2), reduction="median")
