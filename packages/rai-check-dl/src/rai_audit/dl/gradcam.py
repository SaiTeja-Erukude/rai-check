from __future__ import annotations

import importlib

import numpy as np
from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity

_GRADCAM_STANDARDS = ["EU-AI-ACT-ART-15", "NIST-AI-RMF-MEASURE-2.6"]


def pytorch_grad_cam(model, inputs, target_layer, class_indices=None) -> np.ndarray:
    """Generate Grad-CAM heatmaps using PyTorch forward and backward hooks."""
    torch = _require_framework("torch", "torch")
    captured = {}

    def capture_activations(_module, _inputs, output):
        captured["activations"] = output

    def capture_gradients(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(capture_activations)
    backward_handle = target_layer.register_full_backward_hook(capture_gradients)
    try:
        outputs = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
        if outputs.ndim != 2:
            raise ValueError("PyTorch model output must have shape (batch, classes)")
        if class_indices is None:
            targets = outputs.argmax(dim=1)
        else:
            targets = torch.as_tensor(class_indices, device=outputs.device)
        model.zero_grad()
        outputs.gather(1, targets.reshape(-1, 1)).sum().backward()
        activations = captured["activations"]
        gradients = captured["gradients"]
        spatial_axes = tuple(range(2, gradients.ndim))
        weights = gradients.mean(dim=spatial_axes, keepdim=True)
        heatmaps = torch.relu((weights * activations).sum(dim=1))
        return _normalize_heatmaps(heatmaps.detach().cpu().numpy())
    finally:
        forward_handle.remove()
        backward_handle.remove()


def tensorflow_grad_cam(model, inputs, target_layer, class_indices=None) -> np.ndarray:
    """Generate Grad-CAM heatmaps using TensorFlow GradientTape."""
    tf = _require_framework("tensorflow", "tensorflow")
    layer = model.get_layer(target_layer) if isinstance(target_layer, str) else target_layer
    grad_model = tf.keras.models.Model(model.inputs, [layer.output, model.output])
    with tf.GradientTape() as tape:
        activations, outputs = grad_model(inputs)
        if len(outputs.shape) != 2:
            raise ValueError("TensorFlow model output must have shape (batch, classes)")
        if class_indices is None:
            targets = tf.argmax(outputs, axis=1, output_type=tf.int32)
        else:
            targets = tf.convert_to_tensor(class_indices, dtype=tf.int32)
        indices = tf.stack([tf.range(tf.shape(outputs)[0]), targets], axis=1)
        scores = tf.gather_nd(outputs, indices)
    gradients = tape.gradient(scores, activations)
    spatial_axes = tuple(range(1, len(gradients.shape) - 1))
    weights = tf.reduce_mean(gradients, axis=spatial_axes, keepdims=True)
    heatmaps = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))
    return _normalize_heatmaps(heatmaps.numpy())


def gradcam_stability_findings(
    baseline_heatmaps,
    transformed_heatmaps,
    *,
    localization_masks=None,
    min_similarity: float = 0.80,
    min_localization_overlap: float = 0.50,
    activation_threshold: float = 0.50,
) -> list[AuditFinding]:
    """Check Grad-CAM stability under transformations and overlap with optional masks."""
    baseline = _normalized_batch(baseline_heatmaps)
    similarities = {}
    for name, values in transformed_heatmaps.items():
        transformed = _normalized_batch(values)
        if transformed.shape != baseline.shape:
            raise ValueError(f"transformed_heatmaps[{name!r}] must match baseline heatmap shape")
        similarities[name] = round(float(np.mean(_cosine_similarity(baseline, transformed))), 4)
    unstable = {name: score for name, score in similarities.items() if score < min_similarity}
    findings = [
        AuditFinding(
            check_id="GRADCAM-STABILITY-001",
            title="Unstable Grad-CAM explanations" if unstable else "Grad-CAM stability",
            severity=Severity.MEDIUM if unstable else Severity.PASSED,
            description=(
                f"{len(unstable)} transformation(s) have explanation similarity below threshold."
            ),
            evidence={
                "similarity_by_transformation": similarities,
                "unstable_transformations": unstable,
                "min_similarity": min_similarity,
            },
            recommendation="Review preprocessing sensitivity and explanation consistency.",
            category="Explainability",
            remediation_effort=RemediationEffort.MEDIUM,
            standards_refs=_GRADCAM_STANDARDS,
        )
    ]
    if localization_masks is not None:
        masks = np.asarray(localization_masks).astype(bool)
        if masks.shape != baseline.shape:
            raise ValueError("localization_masks must match baseline heatmap shape")
        active = baseline >= activation_threshold
        overlap = [
            float(np.logical_and(cam, mask).sum() / cam.sum()) if cam.sum() else 0.0
            for cam, mask in zip(active, masks)
        ]
        mean_overlap = float(np.mean(overlap))
        findings.append(
            AuditFinding(
                check_id="GRADCAM-LOCALIZATION-001",
                title=(
                    "Grad-CAM localization overlap is low"
                    if mean_overlap < min_localization_overlap
                    else "Grad-CAM localization overlap"
                ),
                severity=Severity.MEDIUM
                if mean_overlap < min_localization_overlap
                else Severity.PASSED,
                description=(
                    f"Mean explanation overlap with localization masks is {mean_overlap:.3f}."
                ),
                evidence={
                    "mean_localization_overlap": round(mean_overlap, 4),
                    "min_localization_overlap": min_localization_overlap,
                    "activation_threshold": activation_threshold,
                },
                recommendation="Review whether model attention aligns with expected regions.",
                category="Explainability",
                standards_refs=_GRADCAM_STANDARDS,
            )
        )
    return findings


def _normalize_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmaps, dtype=float)
    if values.ndim < 2:
        raise ValueError("heatmaps must have a batch dimension and at least one spatial dimension")
    flat = values.reshape(len(values), -1)
    maximums = flat.max(axis=1).reshape((len(values),) + (1,) * (values.ndim - 1))
    return np.divide(values, maximums, out=np.zeros_like(values), where=maximums > 0)


def _normalized_batch(heatmaps) -> np.ndarray:
    return _normalize_heatmaps(np.asarray(heatmaps, dtype=float))


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = first.reshape(len(first), -1)
    right = second.reshape(len(second), -1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.ones(len(left), dtype=float),
        where=denominator > 0,
    )


def _require_framework(module_name: str, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{module_name} is required for this Grad-CAM backend. "
            f"Install rai-check-dl[{extra_name}]."
        ) from exc
