CC = gcc
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra
LDFLAGS = -lm

SRC_DIR = src
BENCH_DIR = bench
BUILD_DIR = build

# Source files
SOLVER_SRC = $(SRC_DIR)/solver.c
SOLVER_HDR = $(SRC_DIR)/solver.h $(SRC_DIR)/hand_eval.h

# Targets
.PHONY: all clean bench test

all: $(BUILD_DIR)/bench_river

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/bench_river: $(BENCH_DIR)/bench_river.c $(SOLVER_SRC) $(SOLVER_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(BENCH_DIR)/bench_river.c $(SOLVER_SRC) -o $@ $(LDFLAGS)

bench: $(BUILD_DIR)/bench_river
	./$(BUILD_DIR)/bench_river

clean:
	rm -rf $(BUILD_DIR)
