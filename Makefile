# smshift - runtime SM re-allocation for NVIDIA Hopper.
# Requires CUDA (tested with 12.6) and the driver library (libcuda).
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH ?= sm_90
NVFLAGS ?= -O2 -arch=$(ARCH) -Iinclude
BUILD := build

.PHONY: all lib test tools examples check clean

all: lib test tools examples

lib: $(BUILD)/libsmshift.so $(BUILD)/libsmshift.a

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/smshift.o: src/smshift.cu include/smshift.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -c $< -o $@

$(BUILD)/libsmshift.so: $(BUILD)/smshift.o
	$(NVCC) -shared -o $@ $< -lcuda

$(BUILD)/libsmshift.a: $(BUILD)/smshift.o
	ar rcs $@ $<

test: $(BUILD)/test_smshift
$(BUILD)/test_smshift: tests/test_smshift.cu $(BUILD)/libsmshift.a
	$(NVCC) $(NVFLAGS) -o $@ $< $(BUILD)/libsmshift.a -lcuda

tools: $(BUILD)/tpcmap
$(BUILD)/tpcmap: tools/tpcmap.cu $(BUILD)/libsmshift.a
	$(NVCC) $(NVFLAGS) -o $@ $< $(BUILD)/libsmshift.a -lcuda

examples: $(BUILD)/pd_demo
$(BUILD)/pd_demo: examples/pd_demo.cu $(BUILD)/libsmshift.a
	$(NVCC) $(NVFLAGS) -o $@ $< $(BUILD)/libsmshift.a -lcuda

# Runs on the GPU: topology map, then the migration tests.
check: all
	$(BUILD)/tpcmap
	$(BUILD)/test_smshift fence
	$(BUILD)/test_smshift churn

clean:
	rm -rf $(BUILD)
