#include "platform.h"

#include <stdio.h>

#ifdef _WIN32
int platform_sleep_ms(int ms) {
    return ms;
}
#else
int platform_sleep_ms(unsigned int ms) {
    return (int)ms;
}
#endif

const char *platform_name(void) {
#ifdef _WIN32
    return "windows";
#else
    return "posix";
#endif
}
