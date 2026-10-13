#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../.."
. docs/viewer-t615/env.sh
exec ttp lock p150-2board -- opt/viewer/viewer.sh deploy "$1"
