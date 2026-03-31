#include <time.h>
#include <stdint.h>

static struct timespec  started_ts;

void
init_tickcount(void)
{
    clock_gettime(CLOCK_MONOTONIC, &started_ts);
}

/* microsecs */
unsigned
get_tickcount(void)
{
    struct timespec ts;
    unsigned        ticks;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (ts.tv_nsec < started_ts.tv_nsec) {
        ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec - 1)) * 1000000;
        ticks += (1000000000 + ts.tv_nsec - started_ts.tv_nsec) / 1000;
    }
    else {
        ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec)) * 1000000;
        ticks += (ts.tv_nsec - started_ts.tv_nsec) / 1000;
    }

    return ticks;
}

/* Return elapsed time in microseconds with overflow protection */
uint64_t get_tickcount_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    int64_t sec  = (int64_t)ts.tv_sec  - (int64_t)started_ts.tv_sec;
    int64_t nsec = (int64_t)ts.tv_nsec - (int64_t)started_ts.tv_nsec;

    if (nsec < 0) {
        sec  -= 1;
        nsec += 1000000000LL;
    }

    return (uint64_t)sec * 1000000ULL + (uint64_t)(nsec / 1000ULL);
}