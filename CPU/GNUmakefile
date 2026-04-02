AMREX_HOME := $(HOME)/mahesh/softwares/amrex-22.04/build2d
#HYPRE_DIR  := $(HOME)/mahesh/softwares/hypre/src/hypre
#ALGOIM_DIR := $(HOME)/mahesh/softwares/algoim-22.04/algoim

# Project, Src, Programm and obj directories 
#PROJ_DIR    := /home/mahesh/mahesh/code/code_share/incFSI_Multiphase_Axisymmetric_Temperature_Rheology
#PROJ_DIR    := ./../..
PROJ_DIR    := .
PROGRAM_DIR := .
SRC_DIR     := $(PROJ_DIR)/src
SBUILD_DIR  := $(SRC_DIR)/obj
EBUILD_DIR  := ./obj

# find the directories to be in search path and add prefix -I
INC_DIRS := .
#INC_DIRS += $(shell find $(SRC_DIR) -type d)
# the maxdepth will only take 1 sybdirectory hence avoid all in obj deriectory created in dependecy step
# if there are more than 1 subdirectory in Src then either increas -maxdepth 1-> 2 or more or remove it comptetely 
INC_DIRS += $(shell find $(SRC_DIR) -maxdepth 1 -type d)
INC_DIRS += $(AMREX_HOME)/include
# $(HYPRE_DIR)/include $(ALGOIM_DIR)
# $(BLITZ_DIR)/include
INC_FLAG := $(addprefix -I,$(INC_DIRS))

LIB_DIRS := $(AMREX_HOME)/lib
# $(HYPRE_DIR)/lib
# $(BLITZ_DIR)/lib
LIB_FLAG := $(addprefix -L,$(LIB_DIRS))

COMPILE_CPP_FLAGS ?= $(shell awk '/Cflags:/ {$$1=$$2=""; print $$0}' $(AMREX_HOME)/lib/pkgconfig/amrex.pc)
COMPILE_LIB_FLAGS ?= $(shell awk '/Libs:/ {$$1=$$2=""; print $$0}' $(AMREX_HOME)/lib/pkgconfig/amrex.pc)

CFLAGS := $(INC_FLAG) $(COMPILE_CPP_FLAGS)
LFLAGS := $(COMPILE_LIB_FLAGS)

CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -g -Wall

# collect .cpp source and .o from Src and Exec
SSRCS := $(shell find $(SRC_DIR) -name "*.cpp")
ESRCS := $(shell find $(PROGRAM_DIR) -name "*.cpp")
SOBJS_TEMP := $(subst $(PROJ_DIR), $(SBUILD_DIR), $(SSRCS))
SOBJS := $(subst .cpp,.o, $(SOBJS_TEMP))
EOBJS_TEMP := $(subst ./, $(EBUILD_DIR)/, $(ESRCS))
EOBJS := $(subst .cpp,.o, $(EOBJS_TEMP))

EDEPS := $(EOBJS:.o=.d)
SDEPS := $(SOBJS:.o=.d)

obj := $(SOBJS) $(EOBJS)

TARGET := run.exe

all: $(TARGET)

$(TARGET): $(obj)
	$(CXX) -o $@ $(CXXFLAGS) $(CFLAGS) $^ $(LIB_FLAG) $(LFLAGS)
#-lHYPRE -lblitz

# Build step for C++ source in SRC
$(SBUILD_DIR)/%.o: $(PROJ_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) -o $@ -c $(CXXFLAGS) $(CFLAGS) $<

# Build step for C++ source in Exec
$(EBUILD_DIR)/%.o: $(PROGRAM_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) -o $@ -c $(CXXFLAGS) $(CFLAGS) $<

-include $(EDEPS)
-include $(SDEPS)

# Build step for dependency source in SRC
$(SBUILD_DIR)/%.d: $(PROJ_DIR)/%.cpp
	mkdir -p $(dir $@)
	@$(CXX) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

# Build step for dependency source in Exec
$(EBUILD_DIR)/%.d: $(PROGRAM_DIR)/%.cpp
	mkdir -p $(dir $@)
	@$(CXX) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

.PHONY: clean deepclean

clean:
	-rm $(TARGET)

deepclean:clean
	-rm -r $(EBUILD_DIR) $(SBUILD_DIR)

