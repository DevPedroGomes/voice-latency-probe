#!/usr/bin/env bash
# Measures the fake agent at known delays and prints the result.
# Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET, for example a local
# `livekit-server --dev` with ws://127.0.0.1:7880, devkey and secret.
set -euo pipefail

out="${1:-runs}"
mkdir -p "$out"
voice-probe make-tones "$out/tones" --count 8

measure() {
  local end_of_turn_ms="$1" agent_ms="$2" report_every="$3" file="$out/$1-$2.jsonl"
  local room="calibration-$RANDOM"
  voice-probe fake-agent --room "$room" --end-of-turn-ms "$end_of_turn_ms" \
    --agent-ms "$agent_ms" --report-every "$report_every" &
  local agent=$!
  sleep 1
  voice-probe call "$out/tones" --room "$room" -o "$file"
  wait "$agent" || true
  echo
  echo "known delays: end of turn $end_of_turn_ms ms, agent $agent_ms ms"
  voice-probe analyze "$file"
}

measure 300 200 1
measure 500 700 1
measure 800 1500 3
