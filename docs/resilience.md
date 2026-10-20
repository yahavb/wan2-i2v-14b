# Resilience: device health for the serving endpoint

`serve.py` runs Wan2.2-I2V-A14B on 8 ranks across 2 trn2 chips (TP=8, LNC2).
A port-level health check only shows that rank 0's HTTP thread is alive. It stays
green when a chip dies. This page describes the device-level health check and how
to simulate a chip failure.

## What a dead chip looks like

Every rank takes part in every collective: T5 and DiT all-reduces, the VAE tile
all-gather, the barriers. When one chip stops, the other ranks block in the next
collective and never return. The common symptom is a hang, not an error.

## Proof of life

Rank 0 keeps one timestamp, `last_ok`.

1. A finished request updates `last_ok`. It ran collectives on all 8 ranks, so it
   proves every chip is alive.
2. When the server has had no proof for `HEARTBEAT_S`, the main loop runs a
   heartbeat: every rank multiplies a one-element tensor on device and all-reduces it.
   Rank 0 checks the sum equals the world size, then updates `last_ok`.

The heartbeat runs only when needed. A busy server never runs it, because its
requests already prove the chips.

## Why it does not starve requests

1. The heartbeat runs in the same main loop as requests, never alongside one. All
   device work stays on each rank's main thread.
2. It runs only when the request queue is empty. A queued request always goes first.
3. It replaces the idle keep-alive broadcast the loop already sent, so it adds no
   new thread or timer.
4. Its cost is one tiny op and one all-reduce of a few bytes. In the worst case a
   request that arrives mid-heartbeat waits for that.

## When /health fails

`/health` returns 503 when any of these holds:

1. Idle, and `last_ok` is older than `HEARTBEAT_S + HEALTH_GRACE_S`.
2. A request has been running longer than `MAX_REQUEST_S`.
3. A heartbeat returned the wrong sum. This one is sticky until the pod restarts.

`/readiness` applies the same checks, so an unhealthy pod also leaves the Service.
Before warmup ends, `/health` returns 200 and `/readiness` returns 503; the
startupProbe holds liveness off until then.

| Setting | Default | Meaning |
|---|---|---|
| `HEARTBEAT_S` | 60 | idle time without proof before a heartbeat runs |
| `HEALTH_GRACE_S` | 60 | extra time allowed before `/health` fails |
| `MAX_REQUEST_S` | 900 | longest a request may run before `/health` fails |
| `FAULT_INJECTION` | 0 (1 in the deploy file) | enables `/debug/fail` |

Probes in `wan2-i2v-14b-deploy.yaml`:

1. startupProbe: `/readiness` every 30 s, up to 4 h, for download, compile and warmup.
2. livenessProbe: `/health` every 30 s, restart after 3 failures.
3. readinessProbe: `/readiness` every 10 s.

Expected time from a dead chip to a restart, while idle: up to about 2 min for
`/health` to fail, then about 1.5 min of liveness failures.

## Simulating a chip failure

With `FAULT_INJECTION=1`:

```
curl -X POST http://wan2-i2v-14b:8000/debug/fail \
  -H 'Content-Type: application/json' -d '{"rank": 4, "mode": "hang"}'
```

1. `hang`: that rank stops at its next device work, either a heartbeat or a request.
   The other ranks block in the collective, `last_ok` goes stale, and `/health` fails.
   This is the dead-chip case.
2. `wrong`: that rank adds 0 instead of 1 in the heartbeat. The sum is wrong and
   `/health` fails at once. This is the case of a chip that runs but computes badly.

Ranks 0 to 3 are expected to be chip 0 and ranks 4 to 7 chip 1. Check with
`neuron-ls` in the pod.

Killing a worker process (`kill -9`) tests a different path: torchrun tears down
the group, the container exits and Kubernetes restarts it. No probe is involved.

## Not covered

1. Real hardware faults, such as memory errors, cannot be triggered from software
   here. Hardware error counters could be read by a separate exec probe; that
   would detect real faults but not simulate them.
2. A rank that raises an error during a request, instead of hanging, may leave the
   others waiting in a collective. The stale timestamp then catches it.
3. Set `FAULT_INJECTION=0` outside testing.
