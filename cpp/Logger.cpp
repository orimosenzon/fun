#include <iostream> 
#include "Logger.hh" 

Logger* Logger::pLoggerInstance = 0; // this simple initialization is guaranteed to 
                                     // be executed before any global ctor invocation 

Logger* Logger::getInstance() {
  if( !pLoggerInstance ) {
    Lock lock; 
    if( !pLoggerInstance ) 
      pLoggerInstance = new Logger(); 
  } 
  return pLoggerInstance;
}

void Logger::log() {
  std::cout << "loging... loging.. \n"; //..   
}

int main() {
    Logger::getInstance()->log();
    Logger::getInstance()->log();
}
