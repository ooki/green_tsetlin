# Define the C++ compiler
CXX = g++

# Define the output file name
OUTPUT = bitwise

# Define the source file
SOURCE = bitwise.cpp

# Define the compile flags (optional)
CFLAGS = -Wall -std=c++17 -mavx2

# Default rule to build the output file
all: $(OUTPUT)

# Rule to build the output file from the source file
$(OUTPUT): $(SOURCE)
	$(CXX) $(CFLAGS) -o $(OUTPUT) $(SOURCE)

# Rule to run the output file
run:
	./$(OUTPUT)

# Rule to clean up the output file
clean:
	rm -f $(OUTPUT)