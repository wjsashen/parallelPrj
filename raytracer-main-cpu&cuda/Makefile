# Compiler
CXX = g++
CXXFLAGS = -Wall -std=c++17

# Target executable (handle Windows vs. others)
ifeq ($(OS),Windows_NT)
    TARGET = raytracer1d.exe
else
    TARGET = raytracer1d
endif

# Source files
SRCS = raycast.cpp parse.cpp
OBJS = $(SRCS:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Pattern rule to compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Cross-platform clean rule
clean:
	@echo "Cleaning up..."
ifeq ($(OS),Windows_NT)
	del /F /Q $(OBJS) $(TARGET) 2> nul || exit 0
else
	rm -f $(OBJS) $(TARGET)
endif

