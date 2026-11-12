#!/usr/bin/env bash
set -euo pipefail

# Ensure working directory is the project root (supports double-clicking from file managers)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If launched without an interactive terminal (e.g., double-clicked in a desktop file manager),
# re-launch inside a terminal emulator so console logs and tracebacks are visible for debugging.
if [ ! -t 0 ] && [ ! -t 1 ] && [ -z "${ECOBEE_IN_TERMINAL:-}" ]; then
    export ECOBEE_IN_TERMINAL=1
    for term in konsole alacritty x-terminal-emulator gnome-terminal xfce4-terminal mate-terminal tilix kitty xterm; do
        if command -v "$term" >/dev/null 2>&1; then
            case "$term" in
                gnome-terminal)
                    exec "$term" -- "$0" "$@"
                    ;;
                xfce4-terminal|mate-terminal)
                    exec "$term" -x "$0" "$@"
                    ;;
                *)
                    exec "$term" -e "$0" "$@"
                    ;;
            esac
        fi
    done
fi

VENV_DIR=".venv"
PACKAGES=(pandas matplotlib scipy numpy)

# Pause before closing if launched via desktop double-click or if an error occurs
cleanup_on_exit() {
    local exit_code=$?
    if [ $exit_code -ne 0 ] || [ -n "${ECOBEE_IN_TERMINAL:-}" ]; then
        echo ""
        echo "Process exited with status $exit_code."
        if [ -t 0 ]; then
            read -rp "Press Enter to close this console window..." _
        fi
    fi
}
trap cleanup_on_exit EXIT

# Initialize uv virtual environment if missing
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Initializing uv virtual environment in $VENV_DIR..."
    uv venv "$VENV_DIR" --python python3
fi

# Install minimal required packages only if missing from the environment
if ! "$VENV_DIR/bin/python" -c "import pandas, matplotlib, scipy, numpy" >/dev/null 2>&1; then
    echo "Installing required packages (${PACKAGES[*]})..."
    uv pip install --python "$VENV_DIR/bin/python" "${PACKAGES[@]}"
fi

echo "Starting Ecobee Analyzer (console output unbuffered)..."
"$VENV_DIR/bin/python" -u ecobee_viewer.py "$@"
