#!/usr/bin/env bash
set -euo pipefail

python -m miniopd.integration launch --mode full "$@"
