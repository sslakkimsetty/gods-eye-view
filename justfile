# God's Eye View — task runner.
#
# `just dev` is all you need for a stock install.
#
# `just up` additionally brings up the local inference stack (voice LLM +
# STT/TTS on a remote Apple Silicon box, reached over SSH tunnels) before
# starting the dev server. Those recipes are machine-specific infrastructure
# and live in a personal ops justfile outside this repo — point OPS_JUSTFILE
# at yours, or ignore `up` entirely and use `dev`.

HOST := env_var_or_default("HOST", "localhost")
PORT := env_var_or_default("PORT", "4173")

# Personal ops justfile holding the mlx-m2-* / audio-m2-* recipes.
OPS_JUSTFILE := env_var_or_default("GEV_OPS_JUSTFILE", env_var("HOME") / "justfile")

_default:
    @just --list

# Dev server only. Everything a stock install needs.
dev:
    npm run dev -- --host {{HOST}} --port {{PORT}}

# Local inference stack + dev server: voice LLM, STT/TTS, both tunnels, then the app.
up:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{OPS_JUSTFILE}}" ]; then
        echo "No ops justfile at {{OPS_JUSTFILE}}." >&2
        echo "Set GEV_OPS_JUSTFILE, or run 'just dev' for the app alone." >&2
        exit 1
    fi
    ops() { just --justfile "{{OPS_JUSTFILE}}" --working-directory "$HOME" "$@"; }

    echo "==> voice LLM (Magi-02:8881 -> localhost:11452)"
    ops mlx-m2-voice-serve
    ops mlx-m2-voice

    echo "==> audio service (Magi-02:8890 -> localhost:11462)"
    ops audio-m2-serve
    ops audio-m2

    echo "==> dev server on http://{{HOST}}:{{PORT}}"
    npm run dev -- --host {{HOST}} --port {{PORT}}

# Are the tunnels and services actually answering?
status:
    #!/usr/bin/env bash
    for entry in "11452:voice LLM" "11462:STT/TTS"; do
        port="${entry%%:*}"; label="${entry#*:}"
        if nc -z localhost "$port" 2>/dev/null; then
            echo "  up    $label (localhost:$port)"
        else
            echo "  DOWN  $label (localhost:$port)"
        fi
    done
    curl -s -m 5 "http://localhost:11462/health" 2>/dev/null && echo || true

# Stop the remote inference servers. Tunnels are left up; they are cheap.
down:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{OPS_JUSTFILE}}" ]; then echo "No ops justfile at {{OPS_JUSTFILE}}." >&2; exit 1; fi
    just --justfile "{{OPS_JUSTFILE}}" --working-directory "$HOME" mlx-m2-voice-stop
    just --justfile "{{OPS_JUSTFILE}}" --working-directory "$HOME" audio-m2-stop

# Unit tests.
test:
    npm test

# Score voice routing against the local model.
eval prompt="local":
    node scripts/qa-voice-routing-local.mjs --prompt {{prompt}}
