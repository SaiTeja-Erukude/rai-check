import numpy as np
from rai_audit.dl.transformations import (
    adjust_brightness,
    evaluate_transformations,
    horizontal_flip,
    prediction_labels,
)


def test_horizontal_flip_flips_width_axis():
    images = np.array([[[[1], [2]], [[3], [4]]]])
    flipped = horizontal_flip(images)
    np.testing.assert_array_equal(flipped, np.array([[[[2], [1]], [[4], [3]]]]))


def test_brightness_preserves_dtype_and_clips():
    images = np.array([[[[255]]]], dtype=np.uint8)
    adjusted = adjust_brightness(images, factor=2.0)
    assert adjusted.dtype == np.uint8
    assert adjusted.item() == 255


def test_evaluate_transformations_converts_probabilities_to_labels():
    results = evaluate_transformations(
        predictor=lambda images: np.array([[0.1, 0.9], [0.8, 0.2]]),
        images=np.zeros((2, 2, 2, 1)),
        transformations={"identity": lambda images: images},
    )
    np.testing.assert_array_equal(results["identity"], np.array([1, 0]))
    np.testing.assert_array_equal(
        prediction_labels(np.array([[0.1, 0.9], [0.8, 0.2]])),
        np.array([1, 0]),
    )
