# voice-latency-probe

[![CI](https://github.com/DevPedroGomes/voice-latency-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/DevPedroGomes/voice-latency-probe/actions/workflows/ci.yml)

A synthetic caller for LiveKit voice agents. It joins a room like a person would, speaks in real time and measures, on its own clock, how long the agent takes to answer. Then it splits each turn into end of turn, agent and delivery.

I built it because the number my agent reported was not the number a caller felt.

```text
1 calls, 8 turns sent, 0 unanswered, 0 with the agent talking over the caller

milliseconds                                    p50     p90     p95     min     max
caller: voice to voice                  n=8    2400    2410    2420    2400    2431
  not reported                          n=6    2400    2416    2424    2400    2431
  reported by the agent                 n=2    2400    2400    2400    2400    2400

stage: end of turn                      n=8     848     849     850     846     850
stage: reply (agent + delivery)         n=8    1553    1562    1572    1551    1581
  by state: agent (LLM, tools, TTS)     n=8    1500    1500    1500    1497    1500
  by state: delivery                    n=8      54      63      74      52      84

agent reported a total on 2 of 8 turns
agent's own number                      n=2    1500    1500    1500    1500    1500
caller, same turns                      n=2    2400    2400    2400    2400    2400
caller minus agent                      n=2     900     900     900     900     900

small sample (8 turns): p90 and p95 are close to the largest value measured.
```

That is a real run against the fake agent described below, which holds 800 ms before closing the turn and 1500 ms before answering, and reports a total on one turn in three. It is in `examples/sample-run.jsonl`, so `voice-probe analyze examples/sample-run.jsonl` reproduces it with no keys and no server.

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
| Reply | `t_agent_voice - t_final` | Agent plus delivery. Needs no state attribute |
| Agent | `t_speaking - t_final` | LLM, tools and TTS |
| Delivery | `t_agent_voice - t_speaking` | Send buffer, network and jitter buffer |

Details and known errors are in [docs/method.md](docs/method.md).

## How I know the numbers are right

The repo includes an agent with known delays. It has no STT, LLM or TTS and needs no paid key: it waits for the caller's audio to go quiet, holds for exactly the configured times and answers with a tone. CI measures it against a real LiveKit server on every push, and the build fails if the probe does not recover the delays.

Latest results on the GitHub runner, loopback to a local server, 8 turns each:

| Delay set in the agent | End of turn, measured p50 | Agent, measured p50 | Delivery p50 | Voice to voice p50 |
|---|---|---|---|---|
| 300 ms and 200 ms | 360 ms | 200 ms | 60 ms | 620 ms |
| 500 ms and 700 ms | 556 ms | 700 ms | 53 ms | 1309 ms |
| 800 ms and 1500 ms | 848 ms | 1500 ms | 54 ms | 2400 ms |

The agent stage comes back within 2 ms. The total reads about 100 to 120 ms above the sum of the delays, and that offset stays steady: 50 ms is the caller's send queue, and the rest is the agent's own 50 ms queue plus the media path. A real caller's device has buffers of its own, so read this as the floor of the instrument. It matters less when you compare two runs, because it is in both.

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
voice-probe analyze examples/sample-run.jsonl
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
