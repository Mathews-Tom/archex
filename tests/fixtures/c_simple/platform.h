#ifndef PLATFORM_H
#define PLATFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
int platform_sleep_ms(int ms);
#else
int platform_sleep_ms(unsigned int ms);
#endif

const char *platform_name(void);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_H */
