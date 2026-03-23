CC = gcc
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra \
         -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
LDFLAGS = -lm

SRC_DIR = src
BENCH_DIR = bench
BUILD_DIR = build

.PHONY: all clean bench dll test

all: $(BUILD_DIR)/bench_v2 $(BUILD_DIR)/solver_v2.dll

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Main benchmark (solver v2)
$(BUILD_DIR)/bench_v2: $(BENCH_DIR)/bench_v2.c $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

# Cross-validation (solver v1 vs Rust)
$(BUILD_DIR)/cross_validate: $(BENCH_DIR)/cross_validate.c $(SRC_DIR)/solver.c $(SRC_DIR)/solver.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $< $(SRC_DIR)/solver.c -o $@ $(LDFLAGS)

# Shared library (solver v2)
$(BUILD_DIR)/solver_v2.dll: $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -I$(SRC_DIR) $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

# Shared library (solver v1, for Python bindings compatibility)
$(BUILD_DIR)/solver.dll: $(SRC_DIR)/solver.c $(SRC_DIR)/solver.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -I$(SRC_DIR) $(SRC_DIR)/solver.c -o $@ $(LDFLAGS)

dll: $(BUILD_DIR)/solver_v2.dll $(BUILD_DIR)/solver.dll

bench: $(BUILD_DIR)/bench_v2
	./$(BUILD_DIR)/bench_v2

test: dll
	python tests/test_end_to_end.py
	python tests/test_integration.py

clean:
	rm -rf $(BUILD_DIR)
