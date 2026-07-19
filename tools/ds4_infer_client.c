/* ds4-infer-client: standalone reference client for the DS4I INFER protocol.
 *
 * Deliberately shares no code with the server: the frame codec and FNV-1a64
 * hash below are written against INFER_PROTOCOL.md alone, exactly like an
 * out-of-tree client would be, so any disagreement with ds4_infer.c is a
 * protocol or documentation bug surfacing in tests rather than in the field.
 *
 * Usage:
 *   ds4-infer-client --connect {host:port|unix:/path}
 *                    [--prefix-file F | --prefix-hash HEX16]
 *                    [--suffix TEXT | --suffix-file F]
 *                    [--cache-only] [--n-predict N] [--temp F] [--top-p F]
 *                    [--min-p F] [--top-k N] [--seed N]
 *   ds4-infer-client --print-hash [FILE]
 *
 * Generated text goes to stdout; result metadata (result_hash, finish reason,
 * token counts, stored flag) goes to stderr.  Exit 0 iff the server replied
 * with status 0.  --print-hash hashes FILE (or stdin) and exits: it doubles
 * as the FNV test-vector checker for other client implementations. */

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define DS4I_MAGIC 0x44533449u
#define DS4I_MSG_INFER 1u
#define DS4I_MSG_RESULT 2u
#define DS4I_VERSION 1u
#define DS4I_F_CACHE_ONLY 1u
#define DS4I_REQUEST_FIXED_BYTES 56u
#define DS4I_RESULT_FIXED_BYTES 44u
#define DS4I_MAX_FRAME_BYTES (16u * 1024u * 1024u)

static uint64_t fnv1a64(const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static void be32_put(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static uint32_t be32_get(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static uint32_t f32_bits(float f) {
    uint32_t v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

static int write_full(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!fp) {
        fprintf(stderr, "ds4-infer-client: cannot open %s: %s\n",
                path, strerror(errno));
        return NULL;
    }
    size_t cap = 65536, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (len == cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        size_t n = fread(buf + len, 1, cap - len, fp);
        len += n;
        if (n == 0) break;
    }
    if (fp != stdin) fclose(fp);
    char *nb = realloc(buf, len + 1);
    if (nb) buf = nb;
    buf[len] = '\0';
    if (len_out) *len_out = len;
    return buf;
}

static int connect_endpoint(const char *spec) {
    if (!strncmp(spec, "unix:", 5)) {
        const char *path = spec + 5;
        struct sockaddr_un sa;
        if (!path[0] || strlen(path) >= sizeof(sa.sun_path)) {
            fprintf(stderr, "ds4-infer-client: invalid unix socket path\n");
            return -1;
        }
        memset(&sa, 0, sizeof(sa));
        sa.sun_family = AF_UNIX;
        strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0 ||
            connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
            fprintf(stderr, "ds4-infer-client: connect %s: %s\n",
                    path, strerror(errno));
            if (fd >= 0) close(fd);
            return -1;
        }
        return fd;
    }

    const char *colon = strrchr(spec, ':');
    if (!colon || !colon[1]) {
        fprintf(stderr,
                "ds4-infer-client: --connect wants host:port or unix:/path\n");
        return -1;
    }
    char host[256];
    size_t hl = (size_t)(colon - spec);
    if (hl == 0 || hl >= sizeof(host)) return -1;
    memcpy(host, spec, hl);
    host[hl] = '\0';

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    int gai = getaddrinfo(host, colon + 1, &hints, &res);
    if (gai != 0) {
        fprintf(stderr, "ds4-infer-client: %s: %s\n", spec, gai_strerror(gai));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0)
        fprintf(stderr, "ds4-infer-client: connect %s: %s\n",
                spec, strerror(errno));
    return fd;
}

static void usage(FILE *fp) {
    fputs("Usage: ds4-infer-client --connect {host:port|unix:/path}\n"
          "         [--prefix-file F | --prefix-hash HEX16]\n"
          "         [--suffix TEXT | --suffix-file F]\n"
          "         [--cache-only] [--n-predict N] [--temp F] [--top-p F]\n"
          "         [--min-p F] [--top-k N] [--seed N]\n"
          "       ds4-infer-client --print-hash [FILE]\n",
          fp);
}

int main(int argc, char **argv) {
    const char *connect_spec = NULL;
    const char *prefix_file = NULL;
    const char *suffix_text = NULL;
    const char *suffix_file = NULL;
    uint64_t prefix_hash = 0xcbf29ce484222325ull; /* empty prefix */
    bool have_prefix_hash = false;
    bool cache_only = false;
    uint32_t n_predict = 256, top_k = 0;
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.05f;
    uint64_t seed = 0;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *val = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(arg, "--print-hash")) {
            size_t len = 0;
            char *data = read_file(val ? val : "-", &len);
            if (!data) return 1;
            printf("%016llx\n", (unsigned long long)fnv1a64(data, len));
            free(data);
            return 0;
        } else if (!strcmp(arg, "--connect") && val) {
            connect_spec = val;
            i++;
        } else if (!strcmp(arg, "--prefix-file") && val) {
            prefix_file = val;
            i++;
        } else if (!strcmp(arg, "--prefix-hash") && val) {
            char *end = NULL;
            prefix_hash = strtoull(val, &end, 16);
            if (end == val || *end != '\0' || strlen(val) != 16) {
                fprintf(stderr,
                        "ds4-infer-client: --prefix-hash wants 16 hex chars\n");
                return 1;
            }
            have_prefix_hash = true;
            i++;
        } else if (!strcmp(arg, "--suffix") && val) {
            suffix_text = val;
            i++;
        } else if (!strcmp(arg, "--suffix-file") && val) {
            suffix_file = val;
            i++;
        } else if (!strcmp(arg, "--cache-only")) {
            cache_only = true;
        } else if (!strcmp(arg, "--n-predict") && val) {
            n_predict = (uint32_t)strtoul(val, NULL, 10);
            i++;
        } else if (!strcmp(arg, "--temp") && val) {
            temperature = strtof(val, NULL);
            i++;
        } else if (!strcmp(arg, "--top-p") && val) {
            top_p = strtof(val, NULL);
            i++;
        } else if (!strcmp(arg, "--min-p") && val) {
            min_p = strtof(val, NULL);
            i++;
        } else if (!strcmp(arg, "--top-k") && val) {
            top_k = (uint32_t)strtoul(val, NULL, 10);
            i++;
        } else if (!strcmp(arg, "--seed") && val) {
            seed = strtoull(val, NULL, 10);
            i++;
        } else if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "ds4-infer-client: unknown option: %s\n", arg);
            usage(stderr);
            return 1;
        }
    }

    if (!connect_spec) {
        usage(stderr);
        return 1;
    }
    if (prefix_file && have_prefix_hash) {
        fprintf(stderr,
                "ds4-infer-client: use --prefix-file or --prefix-hash, not both\n");
        return 1;
    }
    if (prefix_file) {
        size_t len = 0;
        char *data = read_file(prefix_file, &len);
        if (!data) return 1;
        prefix_hash = fnv1a64(data, len);
        free(data);
    }

    char *suffix_owned = NULL;
    const char *suffix = suffix_text ? suffix_text : "";
    size_t suffix_len = strlen(suffix);
    if (suffix_file) {
        suffix_owned = read_file(suffix_file, &suffix_len);
        if (!suffix_owned) return 1;
        suffix = suffix_owned;
    }
    if (memchr(suffix, '\0', suffix_len)) {
        fprintf(stderr, "ds4-infer-client: suffix contains NUL\n");
        free(suffix_owned);
        return 1;
    }
    if (12ull + DS4I_REQUEST_FIXED_BYTES + suffix_len > DS4I_MAX_FRAME_BYTES) {
        fprintf(stderr, "ds4-infer-client: suffix is too large\n");
        free(suffix_owned);
        return 1;
    }

    int fd = connect_endpoint(connect_spec);
    if (fd < 0) {
        free(suffix_owned);
        return 1;
    }

    /* INFER frame: 12-byte header + 56-byte fixed record + suffix bytes. */
    uint8_t frame[12 + DS4I_REQUEST_FIXED_BYTES];
    be32_put(frame + 0, DS4I_MAGIC);
    be32_put(frame + 4, DS4I_MSG_INFER);
    be32_put(frame + 8, DS4I_REQUEST_FIXED_BYTES + (uint32_t)suffix_len);
    be32_put(frame + 12, DS4I_VERSION);
    be32_put(frame + 16, 0);                              /* request_hi */
    be32_put(frame + 20, (uint32_t)getpid());             /* request_lo */
    be32_put(frame + 24, cache_only ? DS4I_F_CACHE_ONLY : 0);
    be32_put(frame + 28, (uint32_t)(prefix_hash >> 32));
    be32_put(frame + 32, (uint32_t)prefix_hash);
    be32_put(frame + 36, (uint32_t)suffix_len);
    be32_put(frame + 40, cache_only ? 0 : n_predict);
    be32_put(frame + 44, f32_bits(temperature));
    be32_put(frame + 48, top_k);
    be32_put(frame + 52, f32_bits(top_p));
    be32_put(frame + 56, f32_bits(min_p));
    be32_put(frame + 60, (uint32_t)(seed >> 32));
    be32_put(frame + 64, (uint32_t)seed);
    if (write_full(fd, frame, sizeof(frame)) != 0 ||
        (suffix_len && write_full(fd, suffix, suffix_len) != 0)) {
        fprintf(stderr, "ds4-infer-client: send failed: %s\n", strerror(errno));
        free(suffix_owned);
        close(fd);
        return 1;
    }
    free(suffix_owned);

    uint8_t hdr[12];
    if (read_full(fd, hdr, sizeof(hdr)) != 0) {
        fprintf(stderr, "ds4-infer-client: no reply: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    if (be32_get(hdr) != DS4I_MAGIC || be32_get(hdr + 4) != DS4I_MSG_RESULT) {
        fprintf(stderr, "ds4-infer-client: unexpected reply frame\n");
        close(fd);
        return 1;
    }
    uint32_t bytes = be32_get(hdr + 8);
    if (bytes < DS4I_RESULT_FIXED_BYTES || bytes > DS4I_MAX_FRAME_BYTES) {
        fprintf(stderr, "ds4-infer-client: bad result frame size\n");
        close(fd);
        return 1;
    }
    uint8_t fixed[DS4I_RESULT_FIXED_BYTES];
    if (read_full(fd, fixed, sizeof(fixed)) != 0) {
        fprintf(stderr, "ds4-infer-client: truncated result\n");
        close(fd);
        return 1;
    }
    uint64_t result_hash = ((uint64_t)be32_get(fixed + 8) << 32) |
                           be32_get(fixed + 12);
    uint32_t status = be32_get(fixed + 16);
    uint32_t finish = be32_get(fixed + 20);
    uint32_t prompt_tokens = be32_get(fixed + 24);
    uint32_t cached_tokens = be32_get(fixed + 28);
    uint32_t generated_tokens = be32_get(fixed + 32);
    uint32_t stored = be32_get(fixed + 36);
    uint32_t text_bytes = be32_get(fixed + 40);
    if (text_bytes != bytes - DS4I_RESULT_FIXED_BYTES) {
        fprintf(stderr, "ds4-infer-client: result text length mismatch\n");
        close(fd);
        return 1;
    }
    char *text = malloc((size_t)text_bytes + 1);
    if (!text || (text_bytes && read_full(fd, text, text_bytes) != 0)) {
        fprintf(stderr, "ds4-infer-client: truncated result text\n");
        free(text);
        close(fd);
        return 1;
    }
    text[text_bytes] = '\0';
    close(fd);

    if (status != 0) {
        fprintf(stderr, "ds4-infer-client: server error: %s\n", text);
        free(text);
        return 1;
    }
    fwrite(text, 1, text_bytes, stdout);
    free(text);
    const char *finish_name = finish == 0 ? "eos" :
                              finish == 1 ? "length" :
                              finish == 2 ? "cache_only" : "?";
    fprintf(stderr,
            "result_hash=%016llx finish=%s prompt=%u cached=%u generated=%u stored=%u\n",
            (unsigned long long)result_hash, finish_name,
            prompt_tokens, cached_tokens, generated_tokens, stored);
    return 0;
}
