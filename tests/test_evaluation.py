from tools.evaluate_gesture import calculate_metrics


def test_evaluation_metrics_are_measured_from_confusion_counts():
    records = [
        {"label": "CROSSED_ARMS"},
        {"label": "CROSSED_ARMS"},
        {"label": "NOT_CROSSED_ARMS"},
        {"label": "NOT_CROSSED_ARMS"},
    ]
    metrics = calculate_metrics(records, [True, False, True, False], 1.25, 800.0)
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_positive_rate == 0.5
    assert metrics.false_negative_rate == 0.5