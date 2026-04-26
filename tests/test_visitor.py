"""Visitor module unittests."""

import math
from genkit.visitor import CoreMetricsVisitor, TrainVisitor


def test_base_visitor_hooks_are_all_no_ops():
    v = TrainVisitor()
    assert v.on_train_start(target=None, source=None, config={}) is None
    assert v.on_epoch_start() is None
    assert v.on_epoch_end() is None
    assert v.format_epoch_log() == ""
    assert v.get_records() == {}


def test_core_metrics_visitor_format_epoch_log_empty_before_first_epoch():
    assert CoreMetricsVisitor().format_epoch_log() == ""


def test_core_metrics_visitor_format_epoch_log_contains_key_fields():
    v = CoreMetricsVisitor()
    v.on_epoch_end(loss=0.123456, loss_std=0.001, grad_var=1e-4, grad_norm=2e-2)
    log = v.format_epoch_log()
    assert "0.123456" in log
    assert "grad_var" in log
    assert "grad_norm" in log


def test_core_metrics_visitor_none_grad_stored_as_nan():
    v = CoreMetricsVisitor()
    v.on_epoch_end(loss=0.5, loss_std=0.0, grad_var=None, grad_norm=None)
    assert math.isnan(v.grad_variance_epoch[-1])
    assert math.isnan(v.grad_norm_epoch[-1])


def test_core_metrics_visitor_get_records_returns_all_keys():
    v = CoreMetricsVisitor()
    assert set(v.get_records()) == {
        "training_loss", "training_loss_std", "grad_variance_epoch", "grad_norm_epoch"
    }


def test_core_metrics_visitor_accumulates_across_epochs():
    v = CoreMetricsVisitor()
    for i in range(3):
        v.on_epoch_end(loss=float(i), loss_std=0.0, grad_var=0.0, grad_norm=0.0)
    assert v.training_loss == [0.0, 1.0, 2.0]
    assert len(v.grad_variance_epoch) == 3
