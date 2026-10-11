"""Installation diagnostics without importing optional framework libraries."""

from __future__ import annotations

import importlib.metadata

_PACKAGES = {
    "rai-check-core": None,
    "rai-check-ml": None,
    "rai-check-kit": None,
    "rai-check-dl": "dl",
    "rai-check-genai": "genai",
}

# Distribution names for the optional extras declared by the audit packages.
_EXTRAS = {
    "rai-check-ml": {"explainability": ("shap",)},
    "rai-check-dl": {"torch": ("torch", "torchvision"), "tensorflow": ("tensorflow",)},
    "rai-check-genai": {"openai": ("openai",), "anthropic": ("anthropic",)},
}


def _package_status(name: str, install_command: str) -> dict:
    result = {"name": name, "version": None, "error": None, "install_command": install_command}
    try:
        result.update(status="installed", version=importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        result["status"] = "not_installed"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def collect_diagnostics(plugins: list[dict], discovery_errors: list[str]) -> dict:
    """Inspect installed distributions and the CLI's recorded plugin load results."""
    packages = []
    optional_dependencies = []
    errors = list(discovery_errors)
    for name, extra in _PACKAGES.items():
        target = f"rai-check-kit[{extra}]" if extra else name
        package = _package_status(name, f'python -m pip install "{target}"')
        package["required"] = extra is None
        packages.append(package)
        if package["status"] == "error":
            errors.append(f"{name}: {package['error']}")
        elif package["required"] and package["status"] != "installed":
            errors.append(f"Required package {name} is not installed.")
        if package["status"] != "installed":
            continue
        for feature, dependencies in _EXTRAS.get(name, {}).items():
            command = f'python -m pip install "{name}[{feature}]"'
            optional_dependencies.append(
                {
                    "package": name,
                    "extra": feature,
                    "dependencies": [_package_status(dep, command) for dep in dependencies],
                    "install_command": command,
                }
            )
    for plugin in plugins:
        if plugin["status"] == "error":
            errors.append(f"Plugin {plugin['name']}: {plugin['error']}")
    for extra in optional_dependencies:
        for dependency in extra["dependencies"]:
            if dependency["status"] == "error":
                errors.append(f"{dependency['name']}: {dependency['error']}")
    return {
        "ok": not errors,
        "packages": packages,
        "plugins": plugins,
        "optional_dependencies": optional_dependencies,
        "errors": errors,
    }


def render_diagnostics(diagnostics: dict) -> str:
    """Render plain text with copyable installation and repair commands."""
    lines = ["RAI Check installation diagnostics", "", "Packages:"]
    for package in diagnostics["packages"]:
        version = f" ({package['version']})" if package["version"] else ""
        lines.append(f"  {package['name']}: {package['status']}{version}")
        if package["status"] != "installed":
            lines.append(f"    Install: {package['install_command']}")
    lines.extend(["", "Plugins:"])
    if not diagnostics["plugins"]:
        lines.append("  No audit plugins discovered.")
    for plugin in diagnostics["plugins"]:
        lines.append(f"  {plugin['name']}: {plugin['status']} ({plugin['entry_point']})")
        if plugin["error"]:
            lines.append(f"    Error: {plugin['error']}")
            lines.append(f"    Repair: {plugin['install_command']}")
    lines.extend(["", "Optional dependencies (installed audit packages only):"])
    for extra in diagnostics["optional_dependencies"]:
        lines.append(f"  {extra['package']}[{extra['extra']}]:")
        for dep in extra["dependencies"]:
            version = f" ({dep['version']})" if dep["version"] else ""
            lines.append(f"    {dep['name']}: {dep['status']}{version}")
        if any(dep["status"] != "installed" for dep in extra["dependencies"]):
            lines.append(f"    Enable: {extra['install_command']}")
    if diagnostics["errors"]:
        lines.extend(["", "Errors:", *(f"  {error}" for error in diagnostics["errors"])])
    lines.extend(
        [
            "",
            "Status: OK" if diagnostics["ok"] else "Status: installation errors found",
            "Missing optional packages and extras are informational.",
        ]
    )
    return "\n".join(lines)
