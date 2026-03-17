#!/usr/bin/env bash
# sync-upstream.sh — Merge upstream aiming-lab/AutoResearchClaw into our fork
# Runs via cron every 3 hours. Logs to scripts/sync-upstream.log
#
# On conflict: aborts the merge and logs the conflict so a human can resolve it.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/scripts/sync-upstream.log"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"

log() { echo "[$TIMESTAMP] $*" >> "$LOG"; }

cd "$REPO_DIR"

log "=== Upstream sync started ==="

# Ensure upstream remote exists
if ! git remote get-url upstream &>/dev/null; then
    git remote add upstream https://github.com/aiming-lab/AutoResearchClaw.git
    log "Added upstream remote"
fi

# Fetch upstream
if ! git fetch upstream 2>>"$LOG"; then
    log "ERROR: git fetch upstream failed"
    exit 1
fi

# Check if there are new commits
LOCAL_HEAD="$(git rev-parse HEAD)"
UPSTREAM_HEAD="$(git rev-parse upstream/main)"
MERGE_BASE="$(git merge-base HEAD upstream/main)"

if [ "$UPSTREAM_HEAD" = "$LOCAL_HEAD" ] || [ "$UPSTREAM_HEAD" = "$MERGE_BASE" ]; then
    log "Already up to date (upstream=$UPSTREAM_HEAD)"
    exit 0
fi

log "New upstream commits detected (local=$LOCAL_HEAD upstream=$UPSTREAM_HEAD)"

# Stash any uncommitted changes
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -m "sync-upstream auto-stash $TIMESTAMP"
    STASHED=true
    log "Stashed uncommitted changes"
fi

# Attempt merge
if git merge upstream/main --no-edit -m "chore: sync with upstream aiming-lab/AutoResearchClaw" 2>>"$LOG"; then
    log "Merge successful"
    
    # Push to origin
    if git push origin main 2>>"$LOG"; then
        log "Pushed to origin/main"
    else
        log "WARNING: Push to origin failed (may need auth)"
    fi
else
    log "ERROR: Merge conflict detected — aborting merge"
    git merge --abort 2>/dev/null || true
    log "Merge aborted. Manual resolution required."
    log "Run: cd $REPO_DIR && git merge upstream/main"
fi

# Restore stashed changes
if [ "$STASHED" = true ]; then
    git stash pop 2>>"$LOG" || log "WARNING: stash pop had conflicts"
fi

log "=== Upstream sync finished ==="
