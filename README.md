# AI-Engineering-From-Scratch

## Adding a new dependency

1. `uv add some-package`
2. `uv lock`
3. `uv sync`

> **Note:** Steps 2 and 3 (`uv lock` and `uv sync`) are not always both required after `uv add`.
>
> - `uv add some-package` modifies your `pyproject.toml`, updates lockfile, and installs the package, so usually that is all you need.
> - If you manually edit dependencies in `pyproject.toml`, _then_ run `uv lock` to update the lockfile, followed by `uv sync` to install changes into your environment.

## Managing the virtual environment

- To **activate**:

  ```bash
  source .venv/bin/activate
  ```

- To **deactivate**:

  ```bash
  deactivate
  ```
