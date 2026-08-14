#!/usr/bin/env bash
# agent-sync — single-file cross-machine sync for coding agents.
#
# Syncs memory / sessions / skills across machines for coding agents:
#   claude   (Claude Code)        sessions + auto-memory -> snapshot into .claude/sessions/
#   opencode (opencode)           sessions -> export/import into .opencode/sessions/ + `oc` wrapper
#   codex    (Codex CLI) [EXPERIMENTAL] heavy: syncs global ~/.codex state (all projects)
#   dsh      (DeepSeek Harness)   sessions -> snapshot into .dsh/sessions/
#
# Usage (copy this file into your project root, then run):
#   ./setup.sh                              show this help (no args -> help)
#   ./setup.sh claude [opencode] [codex] [dsh]  init any combination of tools, codex under [EXPERIMENTAL]
#   ./setup.sh all                          init claude + opencode + dsh (codex separate)
#   ./setup.sh sync                         collect latest agent data (run BEFORE git commit+push)
#   ./setup.sh oc                           wrapper: launch opencode, then export sessions to git，usually directly use `oc` after set `source ~/.zshrc`
#   ./setup.sh --help                       this help
#
# Idempotent: safe to re-run. Never writes API keys.
#
# PRIVACY: session transcripts record everything you type (including API keys).
# Use a PRIVATE repo for this project while developing; don't push the synced
# .claude/sessions, .opencode/sessions, .codex/sessions or .dsh/sessions to a public repo.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$0" 2>/dev/null || true

info() { printf '==> %s\n' "$*"; }
ok()   { printf '    [ok] %s\n' "$*"; }
warn() { printf '    [!!] %s\n' "$*" >&2; }

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
}

# --- helpers ----------------------------------------------------------------

# The encoded project path Claude Code uses as its per-project storage dir name.
encoded() { printf '%s' "$PROJECT" | sed 's|[^A-Za-z0-9]|-|g'; }

# Copy files from $src into $dst that are missing or newer than the copy in
# $dst (merge by mtime). macOS cp has no -u, so compare mtimes manually.
_merge_newer() {  # $1 = source dir, $2 = dest dir
  local src="$1" dst="$2" f rel
  [ -d "$src" ] || return 0
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    if [ "$f" -nt "$dst/$rel" ]; then
      mkdir -p "$(dirname "$dst/$rel")"
      cp -p "$f" "$dst/$rel"
      _merge_copied=$(( ${_merge_copied:-0} + 1 ))
    fi
  done < <(find "$src" -type f -print0 2>/dev/null)
}

# Snapshot a tool's native/global dir into the project (no symlink — the tool
# keeps working exactly as before). Merges by mtime both ways: newer files win
# in each direction, so a session continued on another machine is picked up and
# a locally-updated session is captured.
snapshot_dir() {  # $1 = native/global dir, $2 = project dir
  local g="$1" p="$2"
  mkdir -p "$g" "$p"
  _merge_newer "$p" "$g"   # project -> native (newer wins)
  _merge_newer "$g" "$p"   # native -> project (newer wins)
  ok "snapshot synced: $p"
}

# Idempotent cwd-aware `oc` wrapper: finds the nearest agent-sync project from
# $PWD (via the .agent-sync marker) and runs its setup.sh oc. No single global
# alias fighting over which project it points to.
write_oc_func() {
  local zshrc="$HOME/.zshrc"
  touch "$zshrc"
  python3 - "$zshrc" <<'PYEOF'
import re, sys
path = sys.argv[1]
s = open(path).read()
s = re.sub(r'^alias oc=.*\n', '', s, flags=re.M)
s = re.sub(r'^# agent-sync: oc alias\n', '', s, flags=re.M)
s = re.sub(r'^# >>> agent-sync: oc >>>\n.*?^# <<< agent-sync: oc <<<\n', '', s, flags=re.M | re.S)
block = (
    '# >>> agent-sync: oc >>>\n'
    'unalias oc 2>/dev/null || true\n'
    'oc() {\n'
    '  local p="$PWD"\n'
    '  while [ "$p" != "/" ]; do\n'
    '    if [ -f "$p/.agent-sync" ]; then\n'
    '      bash "$p/setup.sh" oc "$@"; return $?\n'
    '    fi\n'
    '    p="$(dirname "$p")"\n'
    '  done\n'
    '  echo "oc: not in an agent-sync project (run setup.sh first)" >&2\n'
    '  return 1\n'
    '}\n'
    '# <<< agent-sync: oc <<<\n'
)
s = s.rstrip() + "\n\n" + block
open(path, 'w').write(s)
PYEOF
  ok "wrote cwd-aware oc() to ~/.zshrc  (run 'source ~/.zshrc' to activate)"
}

# Create a dir with a one-line README so git tracks it even when empty.
seed_dir() {  # $1 = dir, $2 = description
  local dir="$1" desc="$2"
  mkdir -p "$dir"
  if [ ! -f "$dir/README.md" ]; then
    printf '# %s (agent-sync)\n\n%s\n' "$(basename "$dir")" "$desc" > "$dir/README.md"
    ok "created $dir/"
  else
    ok "$dir/ already exists"
  fi
}

# --- claude -----------------------------------------------------------------

init_claude() {
  info "claude (Claude Code)"
  seed_dir "$PROJECT/.claude/sessions" "Claude Code session transcripts + auto-memory (snapshot from ~/.claude/projects/<encoded>)."
  seed_dir "$PROJECT/.claude/skills"   "Shared skills; auto-loaded by Claude Code, referenced by opencode via skills.paths."
  seed_dir "$PROJECT/.claude/memory"   "Long-lived project notes; git-tracked."
  snapshot_dir "$HOME/.claude/projects/$(encoded)" "$PROJECT/.claude/sessions"
}

# --- opencode ---------------------------------------------------------------

# Import opencode sessions from the project into the local db (re-import if the
# exported JSON is newer than the local db). Used by both init and sync, so a
# session updated on another machine is pulled back in.
opencode_import() {
  if ! command -v opencode >/dev/null 2>&1; then
    warn "opencode not installed (brew install opencode-ai); skipping import"
    return 0
  fi
  local db="${OPENCODE_DB:-$HOME/.local/share/opencode/opencode.db}" n=0 meta id updated db_updated
  for f in "$PROJECT"/.opencode/sessions/*.json; do
    [ -e "$f" ] || continue
    meta="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));i=d.get("info",{});print(i.get("id",""),i.get("time",{}).get("updated",0))' "$f" 2>/dev/null || true)"
    id="${meta%% *}"; updated="${meta##* }"
    if [ -z "$id" ]; then warn "no id in $(basename "$f")"; continue; fi
    if [ -f "$db" ]; then
      db_updated="$(sqlite3 "$db" "SELECT time_updated FROM session WHERE id='$id';" 2>/dev/null | head -1)"
      if [ -n "$db_updated" ] && [ "$db_updated" -ge "$updated" ] 2>/dev/null; then
        ok "skip up-to-date session $id"; continue
      fi
    fi
    if opencode import "$f" >/dev/null 2>&1; then
      ok "imported session $id"; n=$((n+1))
    else
      warn "import failed: $(basename "$f")"
    fi
  done
  info "  imported $n opencode session(s)"
}

init_opencode() {
  info "opencode"
  seed_dir "$PROJECT/.opencode/sessions" "Exported opencode sessions (JSON). Auto-exported by the oc alias after each run."
  opencode_import
  touch "$PROJECT/.agent-sync"
  write_oc_func
}

# --- codex ------------------------------------------------------------------

init_codex() {
  info "codex (Codex CLI) [EXPERIMENTAL]"
  warn "codex sync is heavy: it copies the global ~/.codex state (all projects'"
  warn "sessions). New 'paginated' sessions may not sync 'continue' content."
  seed_dir "$PROJECT/.codex/sessions"  "Codex CLI sessions for THIS project only (filtered by cwd)."
  seed_dir "$PROJECT/.codex/memories"  "Snapshot of global ~/.codex/memories."
  seed_dir "$PROJECT/.codex/skills"    "Snapshot of global ~/.codex/skills (bundled .system/ gitignored)."

  # memories + skills are global per-user; snapshot them (no symlink, no competition).
  snapshot_dir "$HOME/.codex/memories" "$PROJECT/.codex/memories"
  snapshot_dir "$HOME/.codex/skills"   "$PROJECT/.codex/skills"
  if [ ! -f "$PROJECT/.codex/skills/.gitignore" ]; then
    printf '.system/\n' > "$PROJECT/.codex/skills/.gitignore"
    ok "gitignored .codex/skills/.system/"
  fi

  # sessions: codex stores them globally (not per-project), so filter by cwd
  # and COPY only this project's rollouts into the repo.
  sync_codex_sessions
}

# Print the cwd recorded in a rollout's session_meta line (empty if none).
codex_rollout_cwd() {  # $1 = rollout jsonl path
  python3 - "$1" <<'PYEOF'
import json, sys
cwd = ""
for line in open(sys.argv[1], errors="replace"):
    if '"session_meta"' not in line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "session_meta":
        cwd = d.get("payload", {}).get("cwd") or ""
        break
print(cwd)
PYEOF
}

sync_codex_sessions() {
  local src="$HOME/.codex/sessions" dst="$PROJECT/.codex/sessions" n=0 cwd rel
  if [ -L "$src" ]; then
    warn "~/.codex/sessions is a symlink (left by an old agent-sync); restore it, then re-run"
    return 0
  fi
  mkdir -p "$src" "$dst"
  # restore: project -> native, preserving codex's date-based layout
  # (YYYY/MM/DD/). codex re-indexes by scanning ~/.codex/sessions on startup,
  # so no manual sqlite work is needed.
  _merge_copied=0
  _merge_newer "$dst" "$src"
  # sync: native -> project, preserving the date-based layout. Capture a rollout
  # if it's already known to this project (a session continued on another machine
  # keeps its original cwd), or a brand-new local session whose cwd matches.
  while IFS= read -r f; do
    rel="${f#"$src"/}"
    if ! [ -e "$dst/$rel" ]; then
      cwd="$(codex_rollout_cwd "$f")"
      if [ -z "$cwd" ] || ([ "$cwd" != "$PROJECT" ] && [[ "$cwd" != "$PROJECT/"* ]]); then
        continue
      fi
    fi
    if [ "$f" -nt "$dst/$rel" ]; then
      mkdir -p "$(dirname "$dst/$rel")"
      cp -p "$f" "$dst/$rel"
      n=$((n+1))
    fi
  done < <(find -L "$src" -name 'rollout-*.jsonl' 2>/dev/null)
  ok "copied $n codex session(s) for this project (re-run to pick up new ones)"
  if [ "${_merge_copied:-0}" -gt 0 ]; then
    codex_reset_backfill
  fi
}

# codex rebuilds its sqlite index from the rollout JSONL files via a "backfill"
# that only runs when backfill_state.status != 'complete' (triggered on TUI
# startup). Reset it to 'pending' after restoring files, so the next `codex`
# launch re-scans ~/.codex/sessions and indexes the restored sessions.
codex_reset_backfill() {
  local db
  db="$(ls -t "$HOME"/.codex/state_*.sqlite 2>/dev/null | head -1)"
  [ -n "$db" ] || return 0
  sqlite3 "$db" "UPDATE backfill_state SET status='pending', last_watermark=NULL WHERE id=1;" 2>/dev/null || true
  ok "reset codex backfill state (launch codex once to rebuild the index)"
}

# --- dsh (DeepSeek Harness) -------------------------------------------------

# Encode a project path into dsh's per-project session directory key
# (mirrors projectKey() in dsh-session-persistence-jsonl).
dsh_project_key() {
  python3 - "$PROJECT" <<'PYEOF'
import sys
cwd = sys.argv[1]
readable = ""
separator_run = False
for ch in cwd:
    if ch in "/\\:":
        if not separator_run:
            readable += "-"
        separator_run = True
    elif ch != "~" and (("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in "._-"):
        readable += ch
        separator_run = False
    else:
        readable += "~" + format(ord(ch), "04X")
        separator_run = False
key = readable.lstrip("-") or "root"
print("--" + key[:251] + "--")
PYEOF
}

init_dsh() {
  info "dsh (DeepSeek Harness)"
  seed_dir "$PROJECT/.dsh/sessions" "dsh session transcripts (snapshot from ~/.dsh/sessions/<projectKey>)."
  snapshot_dir "$HOME/.dsh/sessions/$(dsh_project_key)" "$PROJECT/.dsh/sessions"
}

# --- sync -------------------------------------------------------------------

# Collect the latest agent data before you commit + push. Snapshots are
# non-invasive copies of each tool's native storage. No git commands run here —
# commit + push manually afterwards.
sync_all() {
  info "sync (collect latest agent data)"
  snapshot_dir "$HOME/.claude/projects/$(encoded)" "$PROJECT/.claude/sessions"
  snapshot_dir "$HOME/.dsh/sessions/$(dsh_project_key)" "$PROJECT/.dsh/sessions"
  snapshot_dir "$HOME/.codex/memories" "$PROJECT/.codex/memories"
  snapshot_dir "$HOME/.codex/skills"   "$PROJECT/.codex/skills"
  sync_codex_sessions
  opencode_import
  echo
  info "now commit + push manually to sync across machines:"
  info "  git add .claude .opencode .codex .dsh && git commit -m 'sync agent data' && git push"
}

# --- oc launcher ------------------------------------------------------------

run_oc() {
  cd "$PROJECT"
  # Single source of truth for the key: env first, then ~/.claude/settings.json.
  if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    ANTHROPIC_AUTH_TOKEN="$(python3 -c 'import json,os;p=os.path.expanduser("~/.claude/settings.json");d=json.load(open(p)) if os.path.exists(p) else {};print(d.get("env",{}).get("ANTHROPIC_AUTH_TOKEN",""))' 2>/dev/null || true)"
  fi
  export ANTHROPIC_AUTH_TOKEN

  case "${1:-}" in
    run|serve|web|attach|export|import|models|session|agent|db|plugin|mcp|acp|stats|github|pr|upgrade|uninstall|completion|debug|providers|auth)
      opencode "$@" || true ;;
    *)
      opencode --auto "$@" || true ;;
  esac

  local db="$HOME/.local/share/opencode/opencode.db" sess_dir="$PROJECT/.opencode/sessions" count=0 id
  mkdir -p "$sess_dir"
  if [ ! -f "$db" ]; then
    warn "no opencode db found; nothing exported"; return 0
  fi
  local ids
  ids="$(sqlite3 "$db" "SELECT id FROM session WHERE (directory = '$PROJECT' OR directory LIKE '$PROJECT/%') AND parent_id IS NULL ORDER BY time_updated DESC;")"
  for id in $ids; do
    if opencode export "$id" 2>/dev/null > "$sess_dir/$id.json" && [ -s "$sess_dir/$id.json" ]; then
      count=$((count+1))
    else
      warn "export failed for session $id"
    fi
  done
  if [ "$count" -gt 0 ]; then
    git add "$sess_dir" 2>/dev/null || true
    if [ "${OC_AUTOCOMMIT:-0}" = "1" ]; then
      git commit -m "sync opencode sessions ($(date +%F))" 2>/dev/null || true
      ok "exported + committed $count session(s)"
    else
      ok "exported $count session(s) -> .opencode/sessions/ (staged; set OC_AUTOCOMMIT=1 to auto-commit)"
    fi
  else
    info "no opencode sessions to export"
  fi
}

# --- dispatch ---------------------------------------------------------------

if [ $# -eq 0 ]; then
  usage; exit 0
fi

case "${1:-}" in
  --help|-h|help) usage; exit 0 ;;
  oc) shift; run_oc "$@"; exit 0 ;;
  sync) sync_all; exit 0 ;;
esac

for a in "$@"; do
  case "$a" in
    claude)   init_claude ;;
    opencode) init_opencode ;;
    codex)    init_codex ;;
    dsh)      init_dsh ;;
    all)      init_claude; init_opencode; init_dsh ;;
    *) warn "unknown tool: $a"; usage; exit 1 ;;
  esac
done

echo
info "done. 完成。"
info "usage / 用法:"
info "  claude / codex / dsh  -> 直接运行 (run directly)"
info "  opencode        -> 用 'oc' 运行,不要跑 opencode，需要做一层包装 (use 'oc', NOT bare 'opencode')"
info "  keep this repo PRIVATE while developing / 开发期请用私有仓库 — 待开发完成后删除跨级共享session"
info "workflow / 工作流:"
info "host1 set ./setup.sh claude -> chat -> ./setup.sh sync -> git push"
info "host2 git pull -> set ./setup.sh claude -> chat -> ./setup.sh sync -> git push"
info "host1 git pull -> chat -> ./setup.sh sync -> git push"
info "host2 git pull -> chat -> ./setup.sh sync -> git push"