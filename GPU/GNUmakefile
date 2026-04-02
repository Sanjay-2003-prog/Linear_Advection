# Paths
AMREX_HOME := ../../../amrex/build2d_cuda

# Project directories
PROJ_DIR := .
SRC_DIR := $(PROJ_DIR)
BUILD_DIR := ./obj

# MPI paths
MPI_INC_DIR := /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB_DIR := /usr/lib/x86_64-linux-gnu/openmpi/lib

# Compilers
CXX = g++
NVCC = nvcc

# CUDA architecture
CUDA_ARCH = sm_75 #modify according to the system

# C++ flags for regular compilation
CXXFLAGS = -std=c++17 -O3 -g -fPIC
CXXFLAGS += -DAMREX_USE_GPU -DAMREX_USE_CUDA -DAMREX_USE_MPI

# Add profiling flags
CXXFLAGS += -DAMReX_PROFILING=ON
CXXFLAGS += -DAMReX_TRACE_PROFILING=ON  # Optional: for HTML output
CXXFLAGS += -DAMReX_TINY_PROFILING=ON   # Optional: more detailed

# nvcc flags
NVCCFLAGS = -std=c++17 -O3 -g
NVCCFLAGS += -arch=$(CUDA_ARCH)
NVCCFLAGS += --expt-relaxed-constexpr
NVCCFLAGS += --extended-lambda
NVCCFLAGS += -Xcompiler="-fPIC"
NVCCFLAGS += -DAMREX_USE_GPU -DAMREX_USE_CUDA -DAMREX_USE_MPI

# Include paths
INCLUDES = -I. -I$(MPI_INC_DIR) -I$(AMREX_HOME)/include

# Library paths
LIB_DIRS = $(AMREX_HOME)/lib $(MPI_LIB_DIR) /usr/local/cuda/lib64
LIB_FLAGS = $(addprefix -L,$(LIB_DIRS))

# Libraries - ADD curand AND gfortran
LIBS = -lamrex -lcurand -lcudart -lmpi -lgfortran -ldl -lrt -lpthread -lm

TARGET = run_cuda.exe

# Source files
SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LIB_FLAGS) $(LIBS) $(NVCCFLAGS)

# Compile .cpp files: Use nvcc with -x cu flag
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) -x cu -c -o $@ $(NVCCFLAGS) $(INCLUDES) $<

# Clean
clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)

# Run
run: $(TARGET)
	mpiexec -n 1 ./$(TARGET)

.PHONY: all clean run
