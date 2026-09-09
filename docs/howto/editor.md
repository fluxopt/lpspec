# Editor completion and offline checking

This page sets up key completion and a schema check for model files, in an
editor and in a job with no Python. It assumes you have a model file and an
editor that speaks the YAML language server.

The language ships its YAML surface as a JSON Schema,
[`schema/math-spec.schema.json`](https://github.com/energy-models/math-spec/blob/main/schema/math-spec.schema.json),
generated from the declarations `lps.check` validates against. The steps below
read it from math-spec over the network. A vendored copy takes a path in the
same slot.

## Map the schema in VS Code

Install the
[Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml),
then map the schema per workspace:

```jsonc
// .vscode/settings.json
"yaml.schemas": { "https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json": ["*.model.yaml"] }
```

or per file, with a modeline on its first line:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json
```

You get key completion, hover docs, the closed vocabulary behind `dtype:`,
`domain:`, `sense:` and the rest, and a squiggle on a misspelled key, before
Python runs.

## Check a file without Python

The same schema checks a model from a shell, which is what a pre-commit hook
or a non-Python CI job wants:

```bash
uvx check-jsonschema --schemafile https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json model.yaml
```

The schema validates structure only. `expression:` and `where:` are strings to
it, so the math inside them is checked by `lps.check`
([the verbs](../reference/api.md)), not by the schema.
