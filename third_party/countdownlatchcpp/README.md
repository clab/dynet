CountdownLatch for C++ multi-threaded programming
================================================

**CountdownLatch in C++**

-----------------------
countdownlatch is a C++ library providing similar functionalities as availble with standard Java
CountdownLatch class. It's main usage is: it allows one or more threads to wait until a set of
operations being performed on other threads completes.


**Below is an example regarding how to use the library**

---

```c++
#include <unistd.h>
#include <thread>
#include <vector>
#include <iostream>
#include <countdownlatch.hpp>

void fun(clatch::countdownlatch *cl) {
    cl->await();
    std::cout << "Wait is over " << std::endl;
}

int main() {
    auto cl = new clatch::countdownlatch(10);
    int i = 0;
    std::vector<std::thread*> ts;
    while (i++ < 2) {
        std::thread *t  = new std::thread(fun, cl);
        ts.push_back(t);
    }

    i = 0;
    while (i++ < 10) {
        sleep(1);
        cl->count_down();
    }
    i = 0;
    while (i < 2) {
        ts[i++]->join();
    }
    i = 0;
    while (i < 2) {
        delete ts[i++];
    }
    delete cl;
    return 0;
}

```
