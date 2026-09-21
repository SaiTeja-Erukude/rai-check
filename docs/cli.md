# CLI Reference

All commands are available via `rai-check` (installed by `rai-check-core` or `rai-check-kit`).

## `rai-check init`

Scaffold a starter `audit.yaml` config file.

```
rai-check init [--project NAME] [--output PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | `my-project` | Project name |
| `--output` | `audit.yaml` | Output config path |
| `--sample-data / --no-sample-data` | `--sample-data` | Write a starter `predictions.csv` beside the config |

---

## `rai-check run`

Run an audit from `audit.yaml`, write configured report formats, create an evidence
manifest, and enforce the configured gate.

```
rai-check run [--config PATH] [--enforce-gate / --no-enforce-gate]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `audit.yaml` | YAML audit configuration |
| `--enforce-gate` | `true` | Exit `1` when configured gate conditions fail |

Configured audits support `classification`, `regression`, `drift`, `image`,
`medical`, `scientific`, `llm`, `rag`, `rag-security`, and `agents` audit types
when their packages are installed.

---

## `rai-check report`

Render an HTML, Markdown, JSON, SARIF, or JUnit report from a saved audit JSON.

```
rai-check report AUDIT_RUN.JSON [--format html|markdown|json|sarif|junit] [--output PATH]
```

---

## `rai-check gate`

CI/CD deployment gate. Exits `1` on failure, `0` on pass.

```
rai-check gate AUDIT_RUN.JSON [--fail-on-critical] [--min-score N] [--output-json PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--fail-on-critical` | `true` | Fail if any critical findings exist |
| `--min-score` | `null` | Minimum required heuristic score |
| `--output-json` | — | Write gate result to JSON |

---

## `rai-check diff`

Compare two audit runs and show what changed.

```
rai-check diff RUN_A.JSON RUN_B.JSON [--output-json PATH]
```

---

## `rai-check history`

List past audit runs from the history directory.

```
rai-check history [--directory PATH]
```

Default directory: `.rai-check/history`

---

## `rai-check export model-card`

Export an audit run as a Markdown model card (HuggingFace-compatible).

```
rai-check export model-card AUDIT_RUN.JSON [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `<input>.model-card.md` | Output `.md` path |
| `--model-name` | project name | Display name |
| `--model-version` | — | Semantic version string |
| `--author` | — | Author / team name |
| `--license-id` | `MIT` | SPDX license identifier |
| `--language` | `en` | ISO 639-1 language code |

---

## `rai-check export standards-coverage`

Export mapped evidence, missing evidence, and explicit non-compliance-claim
language for standards references. Repeat `--required-ref` to define a custom
coverage set.

```
rai-check export standards-coverage AUDIT_RUN.JSON [--output coverage.json] [--required-ref REF]
```

---

## `rai-check export history-dashboard`

Export an HTML dashboard with run trends, category regressions, and artifact
links.

```
rai-check export history-dashboard [--directory .rai-check/history] [--output audit-history.html]
```

---

## `rai-check export eu-post-market`

Generate an EU AI Act-oriented post-market monitoring report from persisted audit
history and incident annotations.

```
rai-check export eu-post-market [--directory .rai-check/history] [--output eu-ai-act-post-market.md]
```

---

## `rai-check ml run`

Run a classification or regression audit from the command line (installed by `rai-check-ml`).

```
rai-check ml run --data predictions.csv --target label [OPTIONS]
```

---

## `rai-check genai llm run`

Audit captured LLM or RAG responses from a YAML test suite (installed by `rai-check-genai`).

```
rai-check genai llm run --suite packages/rai-check-genai/examples/llm_audit_suite.yml [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--audit-type` | `llm` | Audit type: `llm`, `rag`, or `rag-security` |
| `--out` | `llm_audit_report.html` | Output report path |
| `--format` | `html` | Report format: `html`, `markdown`, or `json` |
| `--persist` | `true` | Save the run under `.rai-check/history/` |

---

## `rai-check genai agents run`

Audit a captured canonical agent execution trace (installed by `rai-check-genai`).

```
rai-check genai agents run --trace agent-trace.json [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--allowed-tools` | - | Comma-separated tool allowlist |
| `--out` | `agent_audit_report.html` | Output report path |
| `--format` | `html` | Report format: `html`, `markdown`, or `json` |
| `--persist` | `true` | Save the run under `.rai-check/history/` |

---

## `rai-check dl run`

Audit recorded image classification predictions from CSV (installed by `rai-check-dl`).

```
rai-check dl run --data predictions.csv --task image [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `image` | Audit type: `image`, `medical`, or `scientific` |
| `--transformed-prefix` | `transform_` | Prefix for transformed prediction columns |
| `--patient-id` | - | Patient ID column for medical imaging leakage checks |
| `--split` | - | Dataset split column for medical imaging leakage checks |
| `--site` | - | Collection-site column for medical imaging bias checks |
| `--format` | `html` | Report format: `html`, `markdown`, or `json` |
