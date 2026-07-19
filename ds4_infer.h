#ifndef DS4_INFER_H
#define DS4_INFER_H

/* DS4I: the client-facing raw-completion protocol.
 *
 * A remote client sends client-rendered prompt text and names any previously
 * cached prefix by the FNV-1a64 hash of that prefix's bytes; the server
 * tokenizes, prefills, samples, and replies with the generated text plus the
 * hash of the full transcript bytes, which chains into the next request's
 * prefix hash.  Prefix hashes resolve against the disk KV cache only, so the
 * server must run with --kv-cache-hash fnv1a64 (checkpoint filenames are the
 * wire hashes).  Framing mirrors the internal DS4D distributed protocol
 * (12-byte header, u32 big-endian fields, u64 values as hi/lo halves) but
 * uses its own magic so the two protocols can never be cross-connected.
 * See INFER_PROTOCOL.md for the wire contract. */

#include "ds4.h"
#include "ds4_kvstore.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_INFER_MAGIC 0x44533449u /* DS4I */
#define DS4_INFER_MSG_INFER 1u
#define DS4_INFER_MSG_RESULT 2u /* types 3+ reserved (future streaming) */
#define DS4_INFER_VERSION 1u

#define DS4_INFER_F_CACHE_ONLY (1u << 0)
#define DS4_INFER_F_VALID_MASK DS4_INFER_F_CACHE_ONLY

#define DS4_INFER_FINISH_EOS 0u
#define DS4_INFER_FINISH_LENGTH 1u
#define DS4_INFER_FINISH_CACHE_ONLY 2u

/* FNV-1a64 offset basis: the hash of zero bytes, i.e. "no prefix". */
#define DS4_INFER_EMPTY_PREFIX_HASH 0xcbf29ce484222325ull

#define DS4_INFER_MAX_FRAME_BYTES (16u * 1024u * 1024u)

/* Host-order views of the wire records; the codec below converts to and from
 * the big-endian fixed records. */
typedef struct {
    uint32_t version;
    uint64_t request_id;
    uint32_t flags;
    uint64_t prefix_hash;
    uint32_t suffix_bytes;
    uint32_t n_predict;
    float temperature;
    uint32_t top_k;
    float top_p;
    float min_p;
    uint64_t seed; /* 0 = server picks */
} ds4_infer_request;

typedef struct {
    uint64_t request_id;
    uint64_t result_hash;
    uint32_t status; /* 0 ok; nonzero = error, payload is a UTF-8 message */
    uint32_t finish_reason;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t generated_tokens;
    uint32_t stored;
    uint32_t text_bytes;
} ds4_infer_result;

/* Frame I/O.  Returns 1 on success, 0 on clean EOF, -1 on error; validates
 * the DS4I magic. */
int ds4_infer_read_frame_header(int fd, uint32_t *type, uint32_t *bytes,
                                char *err, size_t errlen);

/* Codec.  send_* return 0 on success, -1 on transport error.
 * ds4_infer_recv_request expects the frame header to be consumed already and
 * returns 0 on success, 1 for a protocol-soft error (bad version/flags/NUL;
 * the frame was drained, the connection stays usable, *out carries the echoed
 * request id) and -1 for a hard framing/transport error. */
int ds4_infer_send_request(int fd, const ds4_infer_request *r,
                           const void *suffix);
int ds4_infer_recv_request(int fd, uint32_t frame_bytes,
                           ds4_infer_request *out, char **suffix,
                           char *err, size_t errlen);
int ds4_infer_send_result(int fd, const ds4_infer_result *r, const void *text);
int ds4_infer_send_error(int fd, uint64_t request_id, const char *msg);
/* Returns 1 on success, 0 on clean EOF, -1 on error.  *text is a malloc'd
 * NUL-terminated copy of the payload (generated text or error message). */
int ds4_infer_recv_result(int fd, ds4_infer_result *out, char **text,
                          char *err, size_t errlen);

/* Endpoints: "unix:/path" selects a Unix domain socket, anything else is
 * parsed as host:port TCP ("*" host binds all interfaces). */
bool ds4_infer_endpoint_is_unix(const char *spec);
int ds4_infer_listen_endpoint(const char *spec, char *err, size_t errlen);
int ds4_infer_connect_endpoint(const char *spec, char *err, size_t errlen);
/* Socket options for one INFER connection (TCP: the usual low-latency set;
 * UDS: send timeout + no SIGPIPE).  Never makes the fd nonblocking. */
void ds4_infer_socket_prepare(int fd, bool is_unix);

/* Execution layer.  Runs one parsed INFER request against the live session
 * using only public engine/kvstore APIs; the host wires in the steps that
 * touch its own state around the shared session. */
typedef struct {
    ds4_engine *engine;
    ds4_session *session;
    ds4_kvstore *kv;
    const ds4_kvstore_trailer_hooks *hooks;
    void *ud;
    /* Checkpoint the live session before a by-key load replaces it (may be
     * NULL).  Mirrors the server's store-current-before-evict discipline. */
    void (*store_live_before_evict)(void *ud);
    /* The live session content changed owners; clear any protocol bindings
     * the host keeps about it (may be NULL). */
    void (*session_replaced)(void *ud);
} ds4_infer_ctx;

/* Fills *res and, when res->status == 0, *text_out (malloc'd, NUL-terminated
 * generated text; empty for CACHE_ONLY).  On failure res->status != 0 and err
 * carries the client-visible message.  Returns 0 iff res->status == 0. */
int ds4_infer_job_run(ds4_infer_ctx *ctx, const ds4_infer_request *req,
                      const char *suffix, ds4_infer_result *res,
                      char **text_out, char *err, size_t errlen);

#endif
