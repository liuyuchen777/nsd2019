CXX := g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
SRC = my_matrix.cpp
TARGET = _matrix.cpython-37m-x86_64-linux-gnu.so
TEST = test_my_matrix.py
BENCH = benchmark.py

# MKL
MKLROOT ?= ${HOME}/opt/conda
INCLUDES += -m64 -I${MKLROOT}/include
LDFLAGS += -L${MKLROOT}/lib -Wl,--no-as-needed \
	-lmkl_rt -lpthread -lm -ldl

# pybind11
INCLUDES += `python -m pybind11 --includes`
CXXFLAGS += -shared -fPIC

.PHONY: all test bench format clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $(TARGET) $(LDFLAGS)

format: $(SRC) $(TEST) $(BENCH)
	clang-format -i $(SRC)
	yapf -i $(TEST) $(BENCH)

test: $(TARGET) $(TEST)
	pytest -sv $(TEST)

bench: $(TARGET) $(BENCH)
	python $(BENCH)

clean:
	rm -rf performance.txt
	rm -rf *.so