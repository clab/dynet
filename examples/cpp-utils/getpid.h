// Platform-independent header to allow calls to getpid()
// 
#ifdef _WINDOWS
    #include <process.h>
#else
	#include <unistd.h>
#endif
