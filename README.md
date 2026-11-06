# voice-latency-probe

[![CI](https://github.com/DevPedroGomes/voice-latency-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/DevPedroGomes/voice-latency-probe/actions/workflows/ci.yml)

A synthetic caller for LiveKit voice agents. It joins a room like a person would, speaks in real time and measures, on its own clock, how long the agent takes to answer. Then it splits each turn into end of turn, agent and delivery.

I built it because the number my agent reported was not the number a caller felt.

```text
milliseconds                                    p50     p90     p95     min     max
caller: voice to voice                  n=4    1262    1305    1312    1230    1320
  not reported                          n=1    1320    1320    1320    1320    1320
  reported by the agent                 n=3    1255    1267    1268    1230    1270

stage: end of turn                      n=4     515     534     537     505     540
stage: agent (LLM, tools, TTS)          n=4     705     724     727     690     730
stage: delivery (network, jitter)       n=4      42      48      49      35      50

agent reported a total on 3 of 4 turns
agent's own number                      n=3     700     708     709     690     710
caller, same turns                      n=3    1255    1267    1268    1230    1270
caller minus agent                      n=3     555     559     560     540     560
```

That output comes from the fixture in `tests/fixtures/run.jsonl`, so you can reproduce it with no keys and no server.

## Why measure from the caller

Metrics inside a voice stack start where that stack can see. In the LiveKit Agents 1.8 deployment this tool came from, with the STT deciding the end of turn, `e2e_latency` started at the STT's end-of-turn event. The wait for that event was left out. Turns spoken with `session.say`, or answered by a rule with no model, reported nothing, so the slowest and the fastest turns could both be missing from the dashboard.

A caller has neither blind spot. It knows when it stopped talking and when it first heard an answer. Both instants come from one monotonic clock in one process, so there is no clock to synchronize.

## What it measures

| Instant | Meaning |
|---|---|
| `t_speech_end` | The last voiced frame the caller sent. Trailing silence in the WAV does not count |
| `t_final` | The caller's final transcript came back on `lk.transcription`: the agent closed the turn |
| `t_speaking` | The agent's `lk.agent.state` attribute became `speaking` |
| `t_agent_voice` | The first frame received from the agent with voice in it |

| Number | Formula | Where the time goes |
|---|---|---|
| Voice to voice | `t_agent_voice - t_speech_end` | What the caller feels |
| End of turn | `t_final - t_speech_end` | VAD or STT deciding the person stopped, plus the path to the worker and back |
| Agent | `t_speaking - t_final` | LLM, tools and TTS |
| Delivery | `t_agent_voice - t_speaking` | Network and jitter buffer |

Details and known errors are in [docs/method.md](docs/method.md).

## How I know the numbers are right

The repo includes an agent with known delays. It has no STT, LLM or TTS and needs no paid key: it waits for the caller's audio to go quiet, holds for exactly the configured times and answers with a tone. CI measures it against a real LiveKit server on every push, and the build fails if the probe does not recover the delays.

CALIBRATION_TABLE

It also misreports itself on purpose, the way a real stack can: its total leaves out the end of turn, and it can stay silent on some turns. The second calibration test checks that the probe catches both.

## Try it

With a local server and no keys:

```bash
pip install -e .
livekit-server --dev   # in another terminal; or the Docker image
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
scripts/calibrate.sh runs
```

Without a server at all:

```bash
voice-probe analyze tests/fixtures/run.jsonl
```

## Measure your own agent

You need WAV files with real speech, 48 kHz, mono and 16-bit, because your STT has to understand them. On macOS:

```bash
say -o utterances/01.wav --data-format=LEI16@48000 "Hi, what are your opening hours?"
```

Then call the agent. `--calls` repeats the whole script, each call in a new room:

```bash
voice-probe call utterances/ -o runs/before.jsonl --calls 6 --agent-name my-agent
voice-probe analyze runs/before.jsonl
```

Change something, measure again the same way, and compare at the median:

```bash
voice-probe compare runs/before.jsonl runs/after.jsonl
```

Useful options:

- `--agent-name` uses explicit dispatch. `--room-prefix` fits a dispatch rule by prefix.
- `--attr key=value` sets participant attributes your agent reads, such as a language.
- `--reported-field` names the JSON field your agent publishes on the data channel with its own total. The default is `total_ms`. Without it, the agent-versus-caller lines stay empty.
- `--labels rules.json` names kinds of turn from your agent's data packets, so a rule-answered turn and an LLM turn get separate rows.

```json
[{"field": "rule_code", "prefix": "CONFIRM", "label": "rule, no LLM"}]
```

A real call has real effects. If your agent writes leads, books appointments or spends a daily cap, each probe call does that too, so point it at a test project.

## Limits

- **LiveKit only.** It reads LiveKit's attributes and streams. On Vapi or Retell you can reuse the method, not the tool.
- **WebRTC only.** No SIP. Phone audio at 8 kHz tends to make STT and end of turn slower, so a phone number can be worse than these numbers.
- **Synthetic speech is regular.** It has no hesitation and no "uh-huh", which affects end of turn in either direction.
- **It measures first voice, not quality.** A fast wrong answer scores well.
- **Small samples lie at the tail.** With 40 turns, p95 is about the second largest value. The report warns below 20.
- **It stops at the received frame.** It does not include a browser's or phone's output buffer.

## Development

```bash
pip install -e '.[dev]'
ruff check src tests && ruff format --check src tests && mypy
pytest -m "not integration"
pytest -m integration     # needs LIVEKIT_URL and a server
```

MIT licensed.
