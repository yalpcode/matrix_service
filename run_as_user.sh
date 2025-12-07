#!/bin/sh

# Exit on any error and treat unset variables as errors
set -euo

OLD_UID=${1:-${UID:-1000}}
OLD_GID=${2:-${GID:-1000}}
shift 2 || true

getent group $OLD_GID >/dev/null 2>&1 || groupadd --gid $OLD_GID --non-unique user
id -u $OLD_UID >/dev/null 2>&1 || useradd --uid $OLD_UID --gid $OLD_GID --non-unique user

sudo -E -u user "$@"
