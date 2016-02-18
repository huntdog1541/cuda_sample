EXTRA_NVCCFLAGS :=
EXTRA_LDFLAGS :=
EXTRA_CCFLAGS := -g -G

INCLUDES := -I../../common/inc

GCC := g++
NVCC := nvcc -ccbin $(GCC) $(INCLUDES)

NVCCFLAGS := -m64
CFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)


#Generate ASS code for each SM architecture listed in $(SMS)
SMS ?= 11 20 30 35 37 50
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: vecadd

vecadd.o : vecadd.cu
	$(NVCC) $(EXTRA_CCFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

vecadd : vecadd.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

clean:
	rm -f vecadd vecadd.o