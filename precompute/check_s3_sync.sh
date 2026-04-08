#!/bin/bash
# Verify that the S3 source mirror at s3://poker-blueprint-unified/code/ matches
# the local git working tree for the files EC2 launch scripts will compile from.
#
# Why this exists: launch scripts use `aws s3 sync s3://.../code/ $WORKDIR/` to
# fetch source onto a fresh EC2 instance. If you commit a fix to git but forget
# to push the same files to S3, the EC2 instance compiles the OLD version and
# silently produces a bad .bps. This happened on 2026-04-07 with the Bug B fix
# (see STATUS.md "Recent decisions").
#
# This script catches the gap BEFORE you spend EC2 compute on a doomed run.
#
# Usage:
#   bash precompute/check_s3_sync.sh                  # check, exit 1 on mismatch
#   bash precompute/check_s3_sync.sh --auto-fix       # check, upload local → S3 if mismatch
#
# Files checked: the C source + headers + Python files actually used by the
# compile + run pipeline. NOT every file in the repo.

set -euo pipefail

S3_PREFIX="s3://poker-blueprint-unified/code"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO_FIX="${1:-}"

# Files the EC2 launch scripts compile or import. If you add a new source file
# referenced by a launch script, add it here too.
#
# NOTE: src/mccfr_blueprint.h is intentionally OMITTED. The solver-agent
# maintains a parallel gitignored copy at src/mccfr_blueprint_s3.h with extra
# function declarations (e.g. bp_save_turn_centroids) that don't yet have
# corresponding implementations in the gitted .c. The S3 mirror's .h reflects
# the _s3 version, not the gitted version. The compile succeeds because
# undefined function symbols in a -shared build are resolved at runtime, and
# the export pipeline doesn't call those functions. This is a known divergence
# documented in STATUS.md Bug F. Remove this exception when the
# two-file setup is collapsed to a single canonical version.
FILES=(
    "src/mccfr_blueprint.c"
    "src/card_abstraction.c"
    "src/card_abstraction.h"
    "src/hand_eval.h"
    "precompute/export_v2.py"
    "precompute/blueprint_worker_unified.py"
    "python/blueprint_v2.py"
)

mismatches=()
missing_local=()
missing_s3=()

echo "Checking S3 sync for $S3_PREFIX/..."
echo

for f in "${FILES[@]}"; do
    local_path="$REPO_ROOT/$f"
    s3_path="$S3_PREFIX/$f"

    if [ ! -f "$local_path" ]; then
        missing_local+=("$f")
        echo "  ?  $f  (not found locally)"
        continue
    fi

    # Read local file via git's object store to bypass Windows CRLF conversion.
    # On Linux this is the same content; on Windows git's autocrlf may have
    # converted line endings on checkout, so we want git's canonical version.
    local_tmp="$(mktemp)"
    if (cd "$REPO_ROOT" && git show "HEAD:$f" > "$local_tmp" 2>/dev/null); then
        local_size=$(wc -c < "$local_tmp")
    else
        # Fall back to working tree if not in git (uncommitted file)
        cp "$local_path" "$local_tmp"
        local_size=$(wc -c < "$local_tmp")
    fi
    local_hash=$(sha256sum "$local_tmp" | awk '{print $1}')

    # Fetch S3 file via head-object to get size, then download for hash if needed
    s3_size=$(aws s3api head-object --bucket poker-blueprint-unified --key "code/$f" --query "ContentLength" --output text 2>/dev/null || echo "MISSING")

    if [ "$s3_size" = "MISSING" ]; then
        missing_s3+=("$f")
        echo "  ?  $f  (not on S3)"
        rm -f "$local_tmp"
        continue
    fi

    if [ "$local_size" != "$s3_size" ]; then
        # Sizes differ — definitely out of sync
        s3_tmp="$(mktemp)"
        aws s3 cp "$s3_path" "$s3_tmp" --quiet
        s3_hash=$(sha256sum "$s3_tmp" | awk '{print $1}')
        rm -f "$s3_tmp"
        mismatches+=("$f")
        echo "  ✗  $f  (local $local_size B, S3 $s3_size B — DIFFER)"
    else
        # Sizes match — check hash to confirm
        s3_tmp="$(mktemp)"
        aws s3 cp "$s3_path" "$s3_tmp" --quiet
        s3_hash=$(sha256sum "$s3_tmp" | awk '{print $1}')
        rm -f "$s3_tmp"
        if [ "$local_hash" = "$s3_hash" ]; then
            echo "  ✓  $f  ($local_size B)"
        else
            mismatches+=("$f")
            echo "  ✗  $f  (sizes match but hashes differ — DIFFER)"
        fi
    fi
    rm -f "$local_tmp"
done

echo

if [ ${#mismatches[@]} -eq 0 ] && [ ${#missing_s3[@]} -eq 0 ] && [ ${#missing_local[@]} -eq 0 ]; then
    echo "All files in sync. Safe to launch."
    exit 0
fi

echo "==================================================================="
echo "  S3 SYNC CHECK FAILED"
echo "==================================================================="
[ ${#mismatches[@]} -gt 0 ] && echo "  Mismatched files: ${mismatches[*]}"
[ ${#missing_s3[@]} -gt 0 ] && echo "  Missing on S3:    ${missing_s3[*]}"
[ ${#missing_local[@]} -gt 0 ] && echo "  Missing locally:  ${missing_local[*]}"
echo
echo "Launching an EC2 instance now would compile the STALE S3 source."

if [ "$AUTO_FIX" = "--auto-fix" ]; then
    echo
    echo "--auto-fix specified, uploading local → S3..."
    for f in "${mismatches[@]}" "${missing_s3[@]}"; do
        local_tmp="$(mktemp)"
        if (cd "$REPO_ROOT" && git show "HEAD:$f" > "$local_tmp" 2>/dev/null); then
            :
        else
            cp "$REPO_ROOT/$f" "$local_tmp"
        fi
        echo "  uploading $f"
        aws s3 cp "$local_tmp" "$S3_PREFIX/$f" --quiet
        rm -f "$local_tmp"
    done
    echo "Done. Re-run without --auto-fix to verify."
    exit 0
fi

echo
echo "To fix: bash precompute/check_s3_sync.sh --auto-fix"
echo "Or manually: aws s3 cp <local_file> $S3_PREFIX/<rel_path>"
exit 1
