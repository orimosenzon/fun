class Logger {
  Logger() {}         // private ctor to prevent creation not via getInstance
  
  static Logger* pLoggerInstance; // a pointer to the single instance of Logger  

public:

  static Logger* getInstance(); // the only way to acquire the uniqe logger 

  void log();
}; 
