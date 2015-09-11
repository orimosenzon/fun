#include "A.h"

extern A a; 

class B {
 public:
  B() { a.foo(); } 
}; 
