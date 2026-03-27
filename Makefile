CC = gcc
CFLAGS = -O2 -Wall -Wextra \
         -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
LDFLAGS = -lm

SRC_DIR = src
BENCH_DIR = bench
BUILD_DIR = build

# Detect OS for shared library extension
ifeq ($(OS),Windows_NT)
  SHLIB_EXT = .dll
  SHLIB_FLAGS = -shared -static
else
  SHLIB_EXT = .so
  SHLIB_FLAGS = -shared -fPIC
endif

.PHONY: all clean bench dll test blueprint

all: blueprint $(BUILD_DIR)/solver_v2$(SHLIB_EXT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Blueprint MCCFR engine (production) ──
# Requires both mccfr_blueprint.c and card_abstraction.c (k-means bucketing)
blueprint: $(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT)

$(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT): $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp -I$(SRC_DIR) $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

# ── Street solver v2 (GPU re-solve support library) ──
$(BUILD_DIR)/solver_v2$(SHLIB_EXT): $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

# ── Street solver v1 (legacy, for Python bindings compatibility) ──
$(BUILD_DIR)/solver$(SHLIB_EXT): $(SRC_DIR)/solver.c $(SRC_DIR)/solver.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/solver.c -o $@ $(LDFLAGS)

dll: blueprint $(BUILD_DIR)/solver_v2$(SHLIB_EXT) $(BUILD_DIR)/solver$(SHLIB_EXT)

# ── Benchmarks ──
$(BUILD_DIR)/bench_v2: $(BENCH_DIR)/bench_v2.c $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static -I$(SRC_DIR) $< $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_phase1: $(BENCH_DIR)/test_phase1.c $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static -I$(SRC_DIR) $< $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

bench: $(BUILD_DIR)/bench_v2
	./$(BUILD_DIR)/bench_v2

test: $(BUILD_DIR)/test_phase1
	./$(BUILD_DIR)/test_phase1

clean:
	rm -rf $(BUILD_DIR)
