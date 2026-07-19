# DS4I: the remote INFER protocol

DS4I is `ds4-server`'s client-facing raw-completion protocol.  A client sends
client-rendered prompt text plus a hash naming any previously cached prefix;
the server tokenizes, prefills, samples server-side, and replies with the
generated text plus the hash of the full transcript bytes, which chains
directly into the next request's prefix hash.  It exists for thin remote
clients (the reference implementation is C; the intended consumers include
out-of-tree Rust clients) that want DwarfStar's disk KV cache addressable by
content hash — including pre-caching a system prompt once and reusing it
across connections and server restarts.

Scope and stance:

- **Raw completion.** The client owns prompt rendering.  There is no chat
  parsing, no tool-call translation, no thinking stripping: the suffix bytes
  are tokenized as-is and sampled text is returned verbatim.
- **No authentication or encryption.**  Use a Unix domain socket (filesystem
  permissions) for local clients, and expose TCP only on trusted networks or
  through a tunnel.  The listener must never face untrusted networks.
- **One request at a time.**  INFER requests share the server's single live
  session and job queue with the HTTP API; a connection may pipeline requests
  sequentially but each is answered before the next is read.

Server requirements:

```sh
./ds4-server -m model.gguf \
  --kv-disk-dir ~/.ds4/server-kv \
  --kv-cache-hash fnv1a64 \
  --listen-infer unix:/tmp/ds4-infer.sock   # or --listen-infer 127.0.0.1:8100
```

`--kv-cache-hash fnv1a64` is mandatory: prefix resolution is a direct lookup
of the checkpoint file named by the hash.

Qwen models are supported, with one deliberate asymmetry.  The chat/HTTP
paths keep their disk KV cache disabled for Qwen because they mix two
provenance classes — template-authored control tokens versus client data that
may spell the same bytes — which a byte-only cache key cannot represent.
INFER has exactly one provenance class: the client renders the whole prompt,
so control-token spellings in the suffix (for example `<|im_start|>`) are
parsed as real control tokens, by contract.  Qwen INFER checkpoints are
marked with a rendered-text provenance flag and are only ever loadable by
exact INFER key; the HTTP paths never see them.  Prompt-injection hygiene for
Qwen transcripts is therefore entirely the client's responsibility — exactly
as it is for the rest of this raw protocol.

## Framing

Every frame is a 12-byte header followed by `bytes` of payload.  All fields
are unsigned 32-bit big-endian; 64-bit values travel as `hi`/`lo` halves
(`value = hi << 32 | lo`).

| offset | field | value |
| --- | --- | --- |
| 0 | magic | `0x44533449` (`"DS4I"`) |
| 4 | type | `1` = INFER (request), `2` = RESULT (response); `3+` reserved |
| 8 | bytes | payload length after the header |

The magic is distinct from the internal distributed protocol's `"DS4D"` on
purpose: connecting either protocol to the other's port fails immediately.
Frames larger than 16 MiB (header excluded) are rejected.

## INFER request (type 1)

Payload: a 56-byte fixed record, then `suffix_bytes` of raw UTF-8 suffix text
(NUL bytes are rejected).

| offset | field | meaning |
| --- | --- | --- |
| 0 | version | protocol version, currently `1`; unknown versions get an error RESULT |
| 4 | request_hi | client-chosen request id, echoed in the RESULT |
| 8 | request_lo | |
| 12 | flags | bit 0 = `CACHE_ONLY`; all other bits must be zero |
| 16 | prefix_hash_hi | FNV-1a64 over the prefix text bytes; the offset basis `0xcbf29ce484222325` means "no prefix" |
| 20 | prefix_hash_lo | |
| 24 | suffix_bytes | length of the suffix text after this record |
| 28 | n_predict | max new tokens; must be nonzero unless `CACHE_ONLY` |
| 32 | temperature_f32 | IEEE-754 float bit pattern (Rust: `f32::from_bits`) |
| 36 | top_k | `0` disables top-k |
| 40 | top_p_f32 | IEEE-754 bits |
| 44 | min_p_f32 | IEEE-754 bits |
| 48 | seed_hi | sampler seed; `0` lets the server pick one |
| 52 | seed_lo | |

Semantics:

1. `prefix_hash` is resolved against the disk KV cache **only**: the server
   loads the checkpoint file named `<prefix_hash as %016x>.kv`.  A miss (or a
   checkpoint written for a different model/quant/context) is the error
   `unknown prefix hash` — the client recovers by resending the full prompt
   as the suffix with the empty-prefix hash, or by re-caching the prefix with
   a `CACHE_ONLY` request.
2. The suffix text is tokenized server-side and appended to the exact token
   history restored from the checkpoint, then prefilled.
3. With `CACHE_ONLY`, generation is skipped: the server checkpoints
   prefix+suffix and replies `finish_reason=cache_only`.  A refused store
   (for example a prompt shorter than `--kv-cache-min-tokens`, default 512)
   is the error `cache-only store refused`.
4. Otherwise the server samples up to `n_predict` tokens with the requested
   sampler parameters, stopping early at EOS or a full context.

## RESULT response (type 2)

Payload: a 44-byte fixed record, then `text_bytes` of payload — the generated
text when `status == 0`, or a UTF-8 error message otherwise.

| offset | field | meaning |
| --- | --- | --- |
| 0 | request_hi | echo of the request id |
| 4 | request_lo | |
| 8 | result_hash_hi | FNV-1a64 over prefix ++ suffix ++ generated text bytes |
| 12 | result_hash_lo | |
| 16 | status | `0` ok; nonzero = error, payload is the message |
| 20 | finish_reason | `0` = eos, `1` = length, `2` = cache_only |
| 24 | prompt_tokens | total effective prompt tokens (prefix + suffix) |
| 28 | cached_tokens | tokens restored from the disk checkpoint |
| 32 | generated_tokens | |
| 36 | stored | `1` = a checkpoint named by `result_hash` was persisted |
| 40 | text_bytes | payload length |

Error messages are stable strings a client may match on: `unknown prefix
hash`, `unsupported infer version`, `invalid infer flags`, `suffix contains
NUL`, `kv cache disabled`, `cache-only store refused`, `context length
exceeded`, and `prefill failed: ...` / `generation failed: ...` with detail.

## Hashing

The hash is 64-bit FNV-1a over **client-visible text bytes** (never token
ids): offset basis `0xcbf29ce484222325`, prime `0x100000001b3`, folding one
byte at a time (`h ^= byte; h *= prime`).  Rendered as exactly 16 lowercase
hex digits wherever it appears as text.  Test vectors:

| input | FNV-1a64 |
| --- | --- |
| `""` | `cbf29ce484222325` |
| `"a"` | `af63dc4c8601ec8c` |
| `"foobar"` | `85944171f73967e8` |

`ds4-infer-client --print-hash FILE` checks an implementation against these.

Domains:

- `prefix_hash` = hash of the prefix text bytes (everything the client sent
  and received before this request, concatenated in order).
- `result_hash` = hash of prefix ++ suffix ++ generated bytes — i.e. the next
  turn's prefix hash.  The client never needs to recompute it, only echo it,
  but because the domain is plain text bytes it *can* compute any prefix hash
  offline (for example, of a known system prompt file).

The EOS token is excluded from the returned text, from `result_hash`, and
from the stored checkpoint, so the transcript a client accumulates is exactly
replayable byte-for-byte.

## Cache lifecycle

Checkpoints live in `--kv-disk-dir` under the server's normal budget and
eviction rules (`--kv-disk-space-mb`, `--kv-cache-min-tokens`, ...), so a
hash can expire: treat `unknown prefix hash` as a normal, recoverable event.
`stored=0` on a successful generation means the reply's `result_hash` will
*not* resolve next turn — resend the full transcript then.  Explicitly
addressed prefixes are never consumed on load, only hit-touched, so a
pre-cached system prompt survives repeated use until evicted by budget
pressure.

A typical session:

```sh
# One-time (or after eviction): cache the system prompt, note the hash.
ds4-infer-client --connect unix:/tmp/ds4-infer.sock \
  --suffix-file sysprompt.txt --cache-only
# stderr: result_hash=9f0e...cafe finish=cache_only ... stored=1

# Each turn: address the cached prefix, send only the new bytes.
ds4-infer-client --connect unix:/tmp/ds4-infer.sock \
  --prefix-hash 9f0e...cafe --suffix-file turn.txt --n-predict 512
# stdout: generated text; stderr: result_hash=<next prefix hash> ...
```

## Versioning and future work

The frame header has no version; the INFER record's `version` field gates
the request semantics, and unknown RESULT `finish_reason`/flag values must be
ignored by clients.  Reserved for later: a `STREAM` request flag with chunked
RESULT frames (type 3+), and MTP speculative decoding on the server side.
