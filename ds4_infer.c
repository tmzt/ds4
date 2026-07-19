/* DS4I client protocol: wire codec, UDS/TCP endpoints, and the execution
 * layer that runs one INFER request against the live session.  See
 * ds4_infer.h for the protocol overview and INFER_PROTOCOL.md for the wire
 * contract.  The host process (ds4-server) owns the listener threads, the
 * job queue, and the live session; everything here is host-agnostic. */

#include "ds4_infer.h"
#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

/* =========================================================================
 * Wire Records
 * ========================================================================= */

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_infer_frame_header;

typedef struct {
    uint32_t version;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t flags;
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t suffix_bytes;
    uint32_t n_predict;
    uint32_t temperature_f32;
    uint32_t top_k;
    uint32_t top_p_f32;
    uint32_t min_p_f32;
    uint32_t seed_hi;
    uint32_t seed_lo;
} ds4_infer_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t status;
    uint32_t finish_reason;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t generated_tokens;
    uint32_t stored;
    uint32_t text_bytes;
} ds4_infer_result_fixed;

static void infer_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static uint64_t infer_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}

static uint32_t infer_f32_bits(float f) {
    uint32_t v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

static float infer_f32_from_bits(uint32_t v) {
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

static void infer_fixed_to_wire(ds4_infer_fixed *w) {
    uint32_t *p = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(uint32_t); i++) p[i] = htonl(p[i]);
}

static void infer_fixed_from_wire(ds4_infer_fixed *w) {
    uint32_t *p = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(uint32_t); i++) p[i] = ntohl(p[i]);
}

static void infer_result_fixed_to_wire(ds4_infer_result_fixed *w) {
    uint32_t *p = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(uint32_t); i++) p[i] = htonl(p[i]);
}

static void infer_result_fixed_from_wire(ds4_infer_result_fixed *w) {
    uint32_t *p = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(uint32_t); i++) p[i] = ntohl(p[i]);
}

/* =========================================================================
 * Frame I/O
 * ========================================================================= */

static int infer_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    ds4_infer_frame_header h = {
        htonl(DS4_INFER_MAGIC),
        htonl(type),
        htonl(bytes),
    };
    return ds4_dist_io_write_full(fd, &h, sizeof(h));
}

int ds4_infer_read_frame_header(int fd, uint32_t *type, uint32_t *bytes,
                                char *err, size_t errlen) {
    ds4_infer_frame_header h;
    int rc = ds4_dist_io_read_full(fd, &h, sizeof(h));
    if (rc < 0 && errlen)
        snprintf(err, errlen, "failed to read frame header: %s", strerror(errno));
    if (rc <= 0) return rc;
    uint32_t magic = ntohl(h.magic);
    if (magic != DS4_INFER_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        return -1;
    }
    *type = ntohl(h.type);
    *bytes = ntohl(h.bytes);
    return 1;
}

static int infer_discard_bytes(int fd, uint32_t bytes) {
    unsigned char buf[4096];
    while (bytes > 0) {
        size_t n = bytes < sizeof(buf) ? bytes : sizeof(buf);
        int rc = ds4_dist_io_read_full(fd, buf, n);
        if (rc <= 0) return -1;
        bytes -= (uint32_t)n;
    }
    return 0;
}

static bool infer_bytes_have_nul(const char *p, size_t len) {
    return memchr(p, '\0', len) != NULL;
}

/* =========================================================================
 * Codec
 * ========================================================================= */

int ds4_infer_send_request(int fd, const ds4_infer_request *r,
                           const void *suffix) {
    if (!r || (r->suffix_bytes != 0 && !suffix)) return -1;
    const uint64_t frame_bytes = sizeof(ds4_infer_fixed) + (uint64_t)r->suffix_bytes;
    if (frame_bytes > DS4_INFER_MAX_FRAME_BYTES) return -1;

    ds4_infer_fixed w = {0};
    w.version = r->version;
    infer_u64_to_halves(r->request_id, &w.request_hi, &w.request_lo);
    w.flags = r->flags;
    infer_u64_to_halves(r->prefix_hash, &w.prefix_hash_hi, &w.prefix_hash_lo);
    w.suffix_bytes = r->suffix_bytes;
    w.n_predict = r->n_predict;
    w.temperature_f32 = infer_f32_bits(r->temperature);
    w.top_k = r->top_k;
    w.top_p_f32 = infer_f32_bits(r->top_p);
    w.min_p_f32 = infer_f32_bits(r->min_p);
    infer_u64_to_halves(r->seed, &w.seed_hi, &w.seed_lo);
    infer_fixed_to_wire(&w);

    if (infer_write_frame_header(fd, DS4_INFER_MSG_INFER,
                                 (uint32_t)frame_bytes) != 0)
        return -1;
    if (ds4_dist_io_write_full(fd, &w, sizeof(w)) != 0) return -1;
    if (r->suffix_bytes &&
        ds4_dist_io_write_full(fd, suffix, r->suffix_bytes) != 0)
        return -1;
    return 0;
}

int ds4_infer_recv_request(int fd, uint32_t frame_bytes,
                           ds4_infer_request *out, char **suffix,
                           char *err, size_t errlen) {
    if (suffix) *suffix = NULL;
    if (errlen) err[0] = '\0';
    memset(out, 0, sizeof(*out));
    if (frame_bytes < sizeof(ds4_infer_fixed) ||
        frame_bytes > DS4_INFER_MAX_FRAME_BYTES) {
        if (errlen) snprintf(err, errlen, "invalid infer frame size %u", frame_bytes);
        return -1;
    }

    ds4_infer_fixed w;
    if (ds4_dist_io_read_full(fd, &w, sizeof(w)) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read infer record");
        return -1;
    }
    infer_fixed_from_wire(&w);

    out->version = w.version;
    out->request_id = infer_u64_from_halves(w.request_hi, w.request_lo);
    out->flags = w.flags;
    out->prefix_hash = infer_u64_from_halves(w.prefix_hash_hi, w.prefix_hash_lo);
    out->suffix_bytes = w.suffix_bytes;
    out->n_predict = w.n_predict;
    out->temperature = infer_f32_from_bits(w.temperature_f32);
    out->top_k = w.top_k;
    out->top_p = infer_f32_from_bits(w.top_p_f32);
    out->min_p = infer_f32_from_bits(w.min_p_f32);
    out->seed = infer_u64_from_halves(w.seed_hi, w.seed_lo);

    const uint32_t body_bytes = frame_bytes - (uint32_t)sizeof(w);
    if (w.suffix_bytes != body_bytes) {
        if (errlen)
            snprintf(err, errlen, "infer suffix length %u does not match frame",
                     w.suffix_bytes);
        return -1;
    }

    /* Protocol-soft failures: drain the body so the connection stays usable
     * and the caller can reply with an error RESULT echoing the request id. */
    if (w.version != DS4_INFER_VERSION) {
        if (infer_discard_bytes(fd, body_bytes) != 0) return -1;
        if (errlen) snprintf(err, errlen, "unsupported infer version");
        return 1;
    }
    if (w.flags & ~DS4_INFER_F_VALID_MASK) {
        if (infer_discard_bytes(fd, body_bytes) != 0) return -1;
        if (errlen) snprintf(err, errlen, "invalid infer flags");
        return 1;
    }

    char *body = malloc((size_t)body_bytes + 1);
    if (!body) {
        if (errlen) snprintf(err, errlen, "out of memory reading infer suffix");
        return -1;
    }
    if (body_bytes && ds4_dist_io_read_full(fd, body, body_bytes) <= 0) {
        free(body);
        if (errlen) snprintf(err, errlen, "failed to read infer suffix");
        return -1;
    }
    body[body_bytes] = '\0';
    if (infer_bytes_have_nul(body, body_bytes)) {
        free(body);
        if (errlen) snprintf(err, errlen, "suffix contains NUL");
        return 1;
    }
    if (suffix) *suffix = body;
    else free(body);
    return 0;
}

static int infer_send_result_frame(int fd, const ds4_infer_result *r,
                                   const void *text) {
    if (!r || (r->text_bytes != 0 && !text)) return -1;
    const uint64_t frame_bytes =
        sizeof(ds4_infer_result_fixed) + (uint64_t)r->text_bytes;
    if (frame_bytes > DS4_INFER_MAX_FRAME_BYTES) return -1;

    ds4_infer_result_fixed w = {0};
    infer_u64_to_halves(r->request_id, &w.request_hi, &w.request_lo);
    infer_u64_to_halves(r->result_hash, &w.result_hash_hi, &w.result_hash_lo);
    w.status = r->status;
    w.finish_reason = r->finish_reason;
    w.prompt_tokens = r->prompt_tokens;
    w.cached_tokens = r->cached_tokens;
    w.generated_tokens = r->generated_tokens;
    w.stored = r->stored;
    w.text_bytes = r->text_bytes;
    infer_result_fixed_to_wire(&w);

    if (infer_write_frame_header(fd, DS4_INFER_MSG_RESULT,
                                 (uint32_t)frame_bytes) != 0)
        return -1;
    if (ds4_dist_io_write_full(fd, &w, sizeof(w)) != 0) return -1;
    if (r->text_bytes &&
        ds4_dist_io_write_full(fd, text, r->text_bytes) != 0)
        return -1;
    return 0;
}

int ds4_infer_send_result(int fd, const ds4_infer_result *r, const void *text) {
    return infer_send_result_frame(fd, r, text);
}

int ds4_infer_send_error(int fd, uint64_t request_id, const char *msg) {
    if (!msg) msg = "infer error";
    size_t len = strlen(msg);
    if (len > DS4_INFER_MAX_FRAME_BYTES - sizeof(ds4_infer_result_fixed))
        len = DS4_INFER_MAX_FRAME_BYTES - sizeof(ds4_infer_result_fixed);
    ds4_infer_result r = {0};
    r.request_id = request_id;
    r.status = 1;
    r.text_bytes = (uint32_t)len;
    return infer_send_result_frame(fd, &r, msg);
}

int ds4_infer_recv_result(int fd, ds4_infer_result *out, char **text,
                          char *err, size_t errlen) {
    if (text) *text = NULL;
    if (errlen) err[0] = '\0';
    memset(out, 0, sizeof(*out));

    uint32_t type = 0, bytes = 0;
    int rc = ds4_infer_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) return rc;
    if (type != DS4_INFER_MSG_RESULT ||
        bytes < sizeof(ds4_infer_result_fixed) ||
        bytes > DS4_INFER_MAX_FRAME_BYTES) {
        if (errlen) snprintf(err, errlen, "unexpected frame type=%u bytes=%u",
                             type, bytes);
        return -1;
    }

    ds4_infer_result_fixed w;
    if (ds4_dist_io_read_full(fd, &w, sizeof(w)) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read result record");
        return -1;
    }
    infer_result_fixed_from_wire(&w);

    out->request_id = infer_u64_from_halves(w.request_hi, w.request_lo);
    out->result_hash = infer_u64_from_halves(w.result_hash_hi, w.result_hash_lo);
    out->status = w.status;
    out->finish_reason = w.finish_reason;
    out->prompt_tokens = w.prompt_tokens;
    out->cached_tokens = w.cached_tokens;
    out->generated_tokens = w.generated_tokens;
    out->stored = w.stored;
    out->text_bytes = w.text_bytes;

    const uint32_t body_bytes = bytes - (uint32_t)sizeof(w);
    if (w.text_bytes != body_bytes) {
        if (errlen) snprintf(err, errlen, "result text length does not match frame");
        return -1;
    }
    char *body = malloc((size_t)body_bytes + 1);
    if (!body) {
        if (errlen) snprintf(err, errlen, "out of memory reading result text");
        return -1;
    }
    if (body_bytes && ds4_dist_io_read_full(fd, body, body_bytes) <= 0) {
        free(body);
        if (errlen) snprintf(err, errlen, "failed to read result text");
        return -1;
    }
    body[body_bytes] = '\0';
    if (text) *text = body;
    else free(body);
    return 1;
}

/* =========================================================================
 * Endpoints
 * ========================================================================= */

bool ds4_infer_endpoint_is_unix(const char *spec) {
    return spec && strncmp(spec, "unix:", 5) == 0;
}

static int infer_listen_unix(const char *path, char *err, size_t errlen) {
    struct sockaddr_un sa;
    if (!path[0] || strlen(path) >= sizeof(sa.sun_path)) {
        if (errlen) snprintf(err, errlen, "invalid unix socket path");
        return -1;
    }
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);

    /* A stale socket file from a crashed server must be removed before bind,
     * but never delete a live server's socket: probe with a connect first. */
    int probe = socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe >= 0) {
        if (connect(probe, (struct sockaddr *)&sa, sizeof(sa)) == 0) {
            close(probe);
            if (errlen) snprintf(err, errlen, "unix socket %s is in use", path);
            return -1;
        }
        close(probe);
    }
    unlink(path);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        if (errlen) snprintf(err, errlen, "socket: %s", strerror(errno));
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0 ||
        listen(fd, 64) != 0) {
        if (errlen) snprintf(err, errlen, "unable to listen on %s: %s",
                             path, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

static bool infer_split_host_port(const char *spec, char *host, size_t hostlen,
                                  int *port) {
    const char *colon = strrchr(spec, ':');
    if (!colon || !colon[1]) return false;
    char *end = NULL;
    long v = strtol(colon + 1, &end, 10);
    if (end == colon + 1 || *end != '\0' || v < 1 || v > 65535) return false;
    size_t hl = (size_t)(colon - spec);
    if (hl >= hostlen) return false;
    memcpy(host, spec, hl);
    host[hl] = '\0';
    *port = (int)v;
    return true;
}

int ds4_infer_listen_endpoint(const char *spec, char *err, size_t errlen) {
    if (!spec || !spec[0]) {
        if (errlen) snprintf(err, errlen, "empty endpoint");
        return -1;
    }
    if (ds4_infer_endpoint_is_unix(spec))
        return infer_listen_unix(spec + 5, err, errlen);
    char host[256];
    int port = 0;
    if (!infer_split_host_port(spec, host, sizeof(host), &port)) {
        if (errlen)
            snprintf(err, errlen,
                     "endpoint must be host:port or unix:/path, got %s", spec);
        return -1;
    }
    const char *bind_host = (!host[0] || !strcmp(host, "*")) ? NULL : host;
    return ds4_dist_io_listen_tcp(bind_host, port, err, errlen);
}

int ds4_infer_connect_endpoint(const char *spec, char *err, size_t errlen) {
    if (!spec || !spec[0]) {
        if (errlen) snprintf(err, errlen, "empty endpoint");
        return -1;
    }
    if (ds4_infer_endpoint_is_unix(spec)) {
        const char *path = spec + 5;
        struct sockaddr_un sa;
        if (!path[0] || strlen(path) >= sizeof(sa.sun_path)) {
            if (errlen) snprintf(err, errlen, "invalid unix socket path");
            return -1;
        }
        memset(&sa, 0, sizeof(sa));
        sa.sun_family = AF_UNIX;
        strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            if (errlen) snprintf(err, errlen, "socket: %s", strerror(errno));
            return -1;
        }
        if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
            if (errlen) snprintf(err, errlen, "connect %s: %s",
                                 path, strerror(errno));
            close(fd);
            return -1;
        }
        ds4_infer_socket_prepare(fd, true);
        return fd;
    }
    char host[256];
    int port = 0;
    if (!infer_split_host_port(spec, host, sizeof(host), &port)) {
        if (errlen)
            snprintf(err, errlen,
                     "endpoint must be host:port or unix:/path, got %s", spec);
        return -1;
    }
    int fd = ds4_dist_io_connect_tcp(host, port, err, errlen);
    if (fd >= 0) ds4_infer_socket_prepare(fd, false);
    return fd;
}

void ds4_infer_socket_prepare(int fd, bool is_unix) {
    if (!is_unix) {
        ds4_dist_io_set_low_latency(fd);
        return;
    }
    /* TCP_NODELAY and keepalive do not apply to AF_UNIX; a send timeout keeps
     * a dead client from wedging the reply path, and SIGPIPE stays fatal-free
     * like the rest of the transport. */
    struct timeval send_tv = {.tv_sec = 60, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_tv, sizeof(send_tv));
    int one = 1;
    (void)one;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

/* =========================================================================
 * Execution
 * ========================================================================= */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} infer_buf;

static bool infer_buf_append(infer_buf *b, const void *data, size_t len) {
    if (len == 0) return true;
    if (b->len + len + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->len + len + 1) cap *= 2;
        char *p = realloc(b->ptr, cap);
        if (!p) return false;
        b->ptr = p;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, data, len);
    b->len += len;
    b->ptr[b->len] = '\0';
    return true;
}

static int infer_fail(ds4_infer_result *res, char *err, size_t errlen,
                      const char *msg) {
    res->status = 1;
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return -1;
}

/* Store the full live timeline as a checkpoint keyed by the combined
 * client-visible text.  Returns true when the checkpoint was persisted. */
static bool infer_store_combined(ds4_infer_ctx *ctx, const char *combined,
                                 char *store_err, size_t store_err_len) {
    const ds4_tokens *live = ds4_session_tokens(ctx->session);
    if (!live || live->len == 0) return false;
    return ds4_kvstore_store_live_prefix_text(ctx->kv, ctx->engine,
                                              ctx->session, live, live->len,
                                              "cold", combined,
                                              DS4_KVSTORE_EXT_INFER_TEXT,
                                              "infer", ctx->hooks,
                                              store_err, store_err_len);
}

int ds4_infer_job_run(ds4_infer_ctx *ctx, const ds4_infer_request *req,
                      const char *suffix, ds4_infer_result *res,
                      char **text_out, char *err, size_t errlen) {
    if (text_out) *text_out = NULL;
    if (err && errlen) err[0] = '\0';
    memset(res, 0, sizeof(*res));
    res->request_id = req->request_id;

    const bool cache_only = (req->flags & DS4_INFER_F_CACHE_ONLY) != 0;
    if (!ctx->kv || !ctx->kv->enabled)
        return infer_fail(res, err, errlen, "kv cache disabled");
    if (ctx->kv->opt.hash_kind != DS4_KVSTORE_HASH_FNV1A64)
        return infer_fail(res, err, errlen,
                          "kv cache is not in fnv1a64 hash mode");
    if (!cache_only && req->n_predict == 0)
        return infer_fail(res, err, errlen, "n_predict must be positive");
    if (!suffix) suffix = "";
    const size_t suffix_len = strlen(suffix);

    char *prefix_text = NULL;
    size_t prefix_len = 0;
    ds4_tokens prompt = {0};

    if (req->prefix_hash == DS4_INFER_EMPTY_PREFIX_HASH) {
        if (suffix_len == 0)
            return infer_fail(res, err, errlen, "empty prompt");
        if (!ds4_tokenize_rendered_chat_checked(ctx->engine, suffix, &prompt)) {
            ds4_tokens_free(&prompt);
            return infer_fail(res, err, errlen,
                              "prefill failed: suffix tokenization");
        }
    } else {
        char key[41];
        snprintf(key, sizeof(key), "%016llx",
                 (unsigned long long)req->prefix_hash);
        if (ctx->store_live_before_evict) ctx->store_live_before_evict(ctx->ud);
        char kerr[160] = {0};
        int rc = ds4_kvstore_try_load_by_key(ctx->kv, ctx->engine, ctx->session,
                                             key, &prefix_text, &prefix_len,
                                             ctx->hooks, kerr, sizeof(kerr));
        if (rc <= 0) {
            free(prefix_text);
            return infer_fail(res, err, errlen, "unknown prefix hash");
        }
        res->cached_tokens = (uint32_t)rc;
        if (!ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                ctx->engine, ds4_session_tokens(ctx->session), suffix,
                &prompt)) {
            free(prefix_text);
            ds4_tokens_free(&prompt);
            ds4_session_invalidate(ctx->session);
            return infer_fail(res, err, errlen,
                              "prefill failed: suffix tokenization");
        }
    }

    const int ctx_size = ds4_session_ctx(ctx->session);
    if (prompt.len >= ctx_size) {
        free(prefix_text);
        ds4_tokens_free(&prompt);
        ds4_session_invalidate(ctx->session);
        return infer_fail(res, err, errlen, "context length exceeded");
    }

    char serr[256] = {0};
    if (ds4_session_sync(ctx->session, &prompt, serr, sizeof(serr)) != 0) {
        free(prefix_text);
        ds4_tokens_free(&prompt);
        res->status = 1;
        if (err && errlen)
            snprintf(err, errlen, "prefill failed: %s",
                     serr[0] ? serr : "unknown error");
        return -1;
    }
    if (ctx->session_replaced) ctx->session_replaced(ctx->ud);
    res->prompt_tokens = (uint32_t)prompt.len;
    ds4_tokens_free(&prompt);

    /* The combined client-visible transcript: prefix bytes ++ suffix bytes ++
     * generated bytes.  Its FNV-1a64 is both the reply's result_hash and, in
     * fnv mode, the filename of the checkpoint stored below — which is what
     * lets the client chain it into the next request's prefix_hash. */
    infer_buf combined = {0};
    infer_buf gen_text = {0};
    bool oom = !infer_buf_append(&combined, prefix_text, prefix_len) ||
               !infer_buf_append(&combined, suffix, suffix_len);
    free(prefix_text);
    prefix_text = NULL;
    if (oom) {
        free(combined.ptr);
        return infer_fail(res, err, errlen, "out of memory");
    }

    if (cache_only) {
        res->finish_reason = DS4_INFER_FINISH_CACHE_ONLY;
    } else {
        uint64_t rng = req->seed ? req->seed :
            ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^
             (uint64_t)clock());
        const int eos = ds4_token_eos(ctx->engine);
        res->finish_reason = DS4_INFER_FINISH_LENGTH;
        while (res->generated_tokens < req->n_predict) {
            const ds4_tokens *live = ds4_session_tokens(ctx->session);
            if (!live || live->len >= ctx_size) break;
            int token = ds4_session_sample(ctx->session, req->temperature,
                                           (int)req->top_k, req->top_p,
                                           req->min_p, &rng);
            if (token == eos) {
                res->finish_reason = DS4_INFER_FINISH_EOS;
                break;
            }
            if (ds4_session_eval(ctx->session, token, serr, sizeof(serr)) != 0) {
                free(combined.ptr);
                free(gen_text.ptr);
                res->status = 1;
                if (err && errlen)
                    snprintf(err, errlen, "generation failed: %s",
                             serr[0] ? serr : "unknown error");
                return -1;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(ctx->engine, token, &piece_len);
            if (!infer_buf_append(&gen_text, piece, piece_len) ||
                !infer_buf_append(&combined, piece, piece_len)) {
                free(combined.ptr);
                free(gen_text.ptr);
                return infer_fail(res, err, errlen, "out of memory");
            }
            res->generated_tokens++;
        }
    }

    res->result_hash = ds4_kvstore_fnv1a64_bytes(combined.ptr ? combined.ptr : "",
                                                 combined.len);

    char store_err[160] = {0};
    res->stored = infer_store_combined(ctx, combined.ptr ? combined.ptr : "",
                                       store_err, sizeof(store_err)) ? 1u : 0u;
    free(combined.ptr);

    if (cache_only && !res->stored) {
        free(gen_text.ptr);
        res->status = 1;
        if (err && errlen)
            snprintf(err, errlen, "cache-only store refused%s%s",
                     store_err[0] ? ": " : "", store_err);
        return -1;
    }

    res->status = 0;
    res->text_bytes = (uint32_t)gen_text.len;
    if (text_out) {
        *text_out = gen_text.ptr ? gen_text.ptr : calloc(1, 1);
    } else {
        free(gen_text.ptr);
    }
    return 0;
}
