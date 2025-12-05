CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread -mavx -mfma
LDFLAGS = -pthread

TARGET = mxfp4_matmul
SRCS = main.cpp matrix.cpp mxfp4.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
