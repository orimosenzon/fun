
all: Logger fiasco

Logger: Logger.cpp Logger.hh 
	g++ -Wall Logger.cpp -o Logger 

fiasco: A.cpp A.h B.cpp B.h 
	g++ -Wall -c A.cpp
	g++ -Wall -c B.cpp
	g++ -Wall B.o A.o -o fiasco

