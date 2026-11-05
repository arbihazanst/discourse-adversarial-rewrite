#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${PYTHON:-}"
if [[ -z "$runner" ]]; then
  if command -v python >/dev/null 2>&1 && python -c "import sys; sys.exit(sys.version_info < (3,10))" 2>/dev/null; then
    runner=python
  else
    runner=python3
  fi
fi
exec "$runner" "$script_dir/scripts/experiment_cli.py" "$@" --profile main
