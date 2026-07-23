# Repository Guidelines

## Project Structure & Module Organization

The root Python files (`moa.py`, `advanced-moa.py`, `bot.py`, and `providers.py`) contain the original MoA examples and shared provider layer. The local Workbench backend lives in `workbench/`; its React 19 and TypeScript UI is in `ui/src/`. Root-level `test_*.py` files cover providers, workflows, API behavior, storage, security, graphs, and lifecycle handling. Browser-level checks and their fake upstream server live in `e2e/`. `assets/` and `docs/` hold images and design notes. `alpaca_eval/`, `FastChat/`, and `FLASK/` are vendored evaluation projects; avoid broad refactors there unless the evaluation harness is the target.

## Build, Test, and Development Commands

Use the repository virtual environment on Windows:

```powershell
pip install -r requirements.txt
.\scripts\start_workbench.ps1
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest test_workbench_graph
cd ui; npm install; npm run build
```

The launcher starts FastAPI on port 8008 and Vite on 5173. `unittest discover` runs the local suite; `tests.py` is an upstream API-dependent harness and is not part of it. `npm run build` performs strict TypeScript checking before creating `ui/dist`. Run `npm run dev` from `ui/` for frontend-only development.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, type hints, `snake_case` functions/modules, and `PascalCase` classes. Keep async resource ownership explicit and use relative imports inside `workbench`. TypeScript uses two-space indentation, strict types, `PascalCase` React components, and `camelCase` functions and variables. No project-wide formatter is configured, so match neighboring code and keep imports grouped and readable.

## Testing Guidelines

Tests use `unittest`, including `IsolatedAsyncioTestCase` and mocks where external model calls are involved. Name modules `test_<area>.py` and methods `test_<behavior>`. Add regression tests beside the closest root test module. Mock OpenAI-compatible clients and redirect `LOCALAPPDATA` to temporary directories; tests must not require live models, API keys, or persistent user data.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects consistent with history, such as `Fix Workbench honesty gaps` or `Add cloud API models`. Keep each commit focused. Pull requests should explain the user-visible change, list verification commands, link relevant issues, and include screenshots for UI changes. Call out schema, API, security-boundary, or provider-routing changes explicitly.

## Security & Configuration

Route all model calls through `providers.py`; never persist API keys. Preserve `workbench/file_safety.py` checks for allowed roots and stale-file detection. Store secrets in environment variables such as `MOA_API_KEY` or `OPENAI_API_KEY`, never in profiles or committed files.
