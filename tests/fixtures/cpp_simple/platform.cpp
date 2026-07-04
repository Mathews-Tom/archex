#include "platform.hpp"

#ifdef _WIN32
void platform_sleep_ms(int ms) {
    (void)ms;
}
#else
void platform_sleep_ms(int ms) {
    (void)ms;
}
#endif

const char* platform_name() {
    return "generic";
}
