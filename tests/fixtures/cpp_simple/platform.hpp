#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
void platform_sleep_ms(int ms);
#else
void platform_sleep_ms(int ms);
#endif

const char* platform_name();

#ifdef __cplusplus
}
#endif
