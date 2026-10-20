
#pragma once

// DEBUGLEVEL: NO_PRINTS, STATUS or ALL
#define DEBUGLEVEL NO_PRINTS
 









// Debuglevels
#define NO_PRINTS 0
#define STATUS 1 // Only print info about status to check how long it will run or check deadlocks
#define ALL 2 // profiling needed!



#if DEBUGLEVEL == 2
#define LOG(...) printf(__VA_ARGS__)
#define LOG_STATUS(...) printf(__VA_ARGS__)
#define LOG_VECTOR(...) aie::print(__VA_ARGS__)
#elif DEBUGLEVEL == 1
#define LOG(...)
#define LOG_STATUS(...) printf(__VA_ARGS__)
#define LOG_VECTOR(...)
#elif DEBUGLEVEL == 0
#define LOG(...) 
#define LOG_STATUS(...)
#define LOG_VECTOR(...)
#define LOG_VECTOR2(...)
#endif

