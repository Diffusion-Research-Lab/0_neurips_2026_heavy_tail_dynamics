"""Tests for standalone loss helpers."""

import math
import pytest
import torch
from genkit.loss import barron_loss


@pytest.mark.parametrize(
    ("alpha", "scale", "expected"),
    [
        (2.0, 1.0, lambda pred, target: 0.5 * (pred - target).square()),
        (0.0, 2.0, lambda pred, target: torch.log1p(0.5 * ((pred - target) / 2.0).square())),
        (-math.inf, 1.5, lambda pred, target: -torch.expm1(-0.5 * ((pred - target) / 1.5).square())),
    ],
)
def test_barron_loss_matches_special_cases(alpha, scale, expected):
    pred = torch.tensor([1.0, 3.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)
    loss = barron_loss(pred, target, alpha=alpha, scale=scale, reduction="none")
    assert torch.allclose(loss, expected(pred, target))


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
