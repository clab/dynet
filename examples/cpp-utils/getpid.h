// Platform-independent header to allow calls to getpid()
// 
#if _WINDOWS
    #include <process.h>
#else
	#include <unistd.h>
#endif
