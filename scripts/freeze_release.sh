#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s --verify-only | <release-tag>\n' "${0##*/}" >&2
    exit 2
fi

repo_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo_root"

require_clean_checkout() {
    local changes
    changes=$(git status --porcelain --untracked-files=normal)
    if [[ -n $changes ]]; then
        printf 'Release tagging requires a clean checkout.\n' >&2
        exit 1
    fi
}

tag=$1
if [[ $tag != --verify-only ]]; then
    if [[ $tag == -* ]] || ! git check-ref-format "refs/tags/$tag"; then
        printf 'Invalid release tag: %s\n' "$tag" >&2
        exit 2
    fi
    require_clean_checkout
    commit=$(git rev-parse --verify HEAD)
    if git show-ref --verify --quiet "refs/tags/$tag"; then
        printf 'Tag already exists: %s\n' "$tag" >&2
        exit 1
    fi
fi

python3 -m pytest -q
(
    cd frontend
    npm test
    npm run build
)

if [[ $tag == --verify-only ]]; then
    printf 'Release verification passed.\n'
    exit 0
fi

require_clean_checkout
if [[ $(git rev-parse --verify HEAD) != "$commit" ]]; then
    printf 'HEAD changed during verification. Re-run from the intended release commit.\n' >&2
    exit 1
fi

git tag -a -m "ASCS Sound Swarm release $tag" -- "$tag" "$commit"
printf 'Created local tag %s at %s.\n' "$tag" "$commit"
