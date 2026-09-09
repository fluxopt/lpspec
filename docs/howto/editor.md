# Editor completion and offline checking

Key completion and a schema check for model files, in an editor and in a job
with no Python. The language ships its YAML surface as a JSON Schema,
[`schema/math-spec.schema.json`](https://github.com/energy-models/math-spec/blob/main/schema/math-spec.schema.json),
read here from math-spec over the network; a vendored copy takes a path in
the same slot.

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

That gives key completion, hover docs, the closed vocabulary behind
`dtype:`, `domain:` and `sense:`, and a squiggle on a misspelled key.

## Check a file without Python

For a pre-commit hook or a non-Python CI job:

```bash
uvx check-jsonschema --schemafile https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json model.yaml
```

The schema validates structure only. `expression:` and `where:` are strings
to it; the math inside them is checked by [`lps.check`](../reference/api.md).
