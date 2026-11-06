# Method

## The caller

The caller publishes a 48 kHz mono track and sends a frame every 10 ms for the whole call, with silence between utterances. An open microphone behaves this way, and an agent's VAD and STT see what they would see from a person.

It waits for the agent to finish before speaking again. "Finished" means the `listening` state with 0.8 s of quiet, or 2.5 s of quiet when the agent publishes no state. A pause inside a sentence must not read as the end of the answer.

An agent that does not greet is fine. The call goes on after `--greeting-timeout`, and the record says `greeted: false`.

## Sources of error

| Source | Size | Direction |
|---|---|---|
| Send queue: `t_speech_end` is taken when the frame enters a 50 ms queue | up to 50 ms | Total reads high |
| Frame size: voice is detected on received frames | about 10 ms | Total reads high |
| Output buffer of a real device: not included | device dependent | Total reads low |
| Voice threshold: RMS 300 on int16 | a quiet start can be missed by a frame or two | Total reads high |

`t_final` can arrive slightly before `t_speech_end` when the agent closes the turn early. It is kept as a negative end of turn, because that is a finding and not a bug. Anything more than 300 ms before the end belongs to an earlier turn and is ignored.

## Turns that cannot be split

A stage stays empty when the agent does not publish what it needs. No final transcript means no end-of-turn and agent numbers. No `lk.agent.state` means no agent and delivery numbers. The voice-to-voice total does not depend on either.

## What the record holds

Each call is one JSON line in `-o`. It holds room name, generated caller identity, timings, state changes and every JSON data packet the agent published. Transcript text is stored only with `--keep-text`.

Data packets are stored as they arrive. If your agent publishes personal data on the data channel, it ends up in the record, so treat records as you would treat your agent's logs.

Records carry a schema version. A version this tool does not know is refused, not guessed.
