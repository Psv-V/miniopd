#!/usr/bin/env bash
set -euo pipefail

python -m miniopd.integration e2e-smoke "$@"
