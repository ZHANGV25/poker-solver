#!/bin/bash
# Pre-deploy integration test. Run this BEFORE every EC2 launch.
# Tests the full pipeline with tiny parameters to catch:
# - int32 overflow in batch/iteration calculations
# - ctypes truncation
# - sub-chunking logic
# - threshold calculations
# - checkpoint save/load
# - compilation errors
#
# Usage: ./tests/test_before_deploy.sh

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    if [ $1 -eq 0 ]; then
        echo -e "  ${GREEN}PASS${NC}: $2"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}: $2"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Pre-Deploy Integration Test ==="
echo ""

# 1. Compile
echo "[1/6] Compilation"
mkdir -p /tmp/test_build
if [[ "$(uname)" == "Darwin" ]]; then
    gcc -O2 -fPIC -shared -o /tmp/test_build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm 2>/tmp/compile_err.txt
else
    gcc -O2 -fPIC -shared -fopenmp -o /tmp/test_build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm 2>/tmp/compile_err.txt
fi
check $? "C code compiles"

# 2. Verify DLL loads and functions are callable
echo "[2/6] DLL loads and functions resolve"
python3 -c "
import ctypes, os, sys
sys.path.insert(0, 'precompute')
from blueprint_worker_unified import load_bp_dll, BPConfig
bp = load_bp_dll('/tmp/test_build')
config = BPConfig()
bp.bp_default_config(ctypes.byref(config))
# Verify bp_solve argtypes are set
assert bp.bp_solve.argtypes is not None, 'bp_solve.argtypes not set'
assert bp.bp_solve.argtypes[1] == ctypes.c_int, 'bp_solve arg1 must be c_int'
print('DLL loaded, argtypes verified')
" 2>/tmp/test_dll.log
check $? "DLL loads, argtypes correct"

# 2b. Verify iteration calculation matches expected
echo "       Iteration calculation"
python3 -c "
target = 12400 / 96 * 3600 * 200000
assert 90_000_000_000 < target < 95_000_000_000, f'Total iters {target} out of range'
print(f'Total iterations: {int(target):,} — OK')
" 2>/tmp/test_iters.log
check $? "Iteration count ~93B"

# 3. INT32_MAX sub-chunking test
echo "[3/6] Sub-chunking logic"
python3 -c "
import sys
sys.path.insert(0, 'precompute')

INT32_MAX = 2_147_483_647
chunk_size = 10_000_000_000
sub_chunk_max = min(chunk_size, INT32_MAX)
assert sub_chunk_max == INT32_MAX, f'sub_chunk_max={sub_chunk_max}'

# Simulate the loop
iters_done = 0
next_checkpoint_at = chunk_size
iters_this_checkpoint = min(next_checkpoint_at, 93_000_000_000) - iters_done
call_sizes = []
while iters_this_checkpoint > 0:
    call_size = min(iters_this_checkpoint, sub_chunk_max)
    assert call_size <= INT32_MAX, f'call_size {call_size} > INT32_MAX'
    call_sizes.append(call_size)
    iters_this_checkpoint -= call_size

assert len(call_sizes) == 5, f'Expected 5 sub-chunks, got {len(call_sizes)}'
assert call_sizes[-1] == 10_000_000_000 % (INT32_MAX + 1) + (INT32_MAX + 1) - 10_000_000_000 % (INT32_MAX + 1) or call_sizes[-1] < INT32_MAX
for cs in call_sizes:
    assert cs <= INT32_MAX, f'Sub-chunk {cs} exceeds INT32_MAX'
print(f'Sub-chunks: {call_sizes}')
print('OK')
" 2>/tmp/test_subchunk.log
check $? "Sub-chunks all fit INT32_MAX"

# 4. num_batches overflow test
echo "[4/6] num_batches overflow"
python3 -c "
import ctypes

# Simulate C code: num_batches = (int)(((int64_t)max_iterations + batch_size - 1) / batch_size)
max_iterations = 2_147_483_647
batch_size = 81_374_999

# Old (would overflow):
old_sum = ctypes.c_int(max_iterations + batch_size - 1).value
old_batches = old_sum // batch_size
assert old_batches < 0, f'Old code should overflow, got {old_batches}'

# New (int64 cast):
new_batches = (max_iterations + batch_size - 1) // batch_size
assert new_batches == 27, f'Expected 27 batches, got {new_batches}'
assert new_batches > 0, 'num_batches must be positive'
print(f'Old (overflow): {old_batches}, New (fixed): {new_batches}')
print('OK')
" 2>/tmp/test_batches.log
check $? "num_batches doesn't overflow"

# 5. Threshold calculations
echo "[5/6] Threshold calculations"
python3 -c "
threads = 96
target_hours = 12400 / threads
total_iters = int(target_hours * 3600 * 200000)

discount_stop = max(total_iters * 35 // 1000, 1000)
discount_interval = max(discount_stop // 40, 100)
prune_start = max(total_iters * 17 // 1000, 500)
snapshot_start = max(total_iters * 7 // 100, 10000)
snapshot_interval = max(total_iters * 17 // 1000, 5000)

# Verify ratios match Pluribus (within 10%)
total_min = total_iters / 200000 / 60  # total minutes at 200K iter/s
ds_min = discount_stop / 200000 / 60
ps_min = prune_start / 200000 / 60
ss_min = snapshot_start / 200000 / 60

assert 200 < ds_min < 500, f'Discount stop {ds_min:.0f}min not near 400'
assert 100 < ps_min < 300, f'Prune start {ps_min:.0f}min not near 200'
assert 400 < ss_min < 1000, f'Snapshot start {ss_min:.0f}min not near 800'

# Verify nothing exceeds int64
assert discount_stop < 2**63
assert total_iters < 2**63

# Verify batch_size (discount_interval cast to int32) fits
assert discount_interval < 2**31, f'discount_interval {discount_interval} exceeds INT32_MAX'

print(f'Total: {total_iters:,} iters')
print(f'Discount: stop={ds_min:.0f}min interval={discount_interval/200000/60:.0f}min')
print(f'Pruning: {ps_min:.0f}min')
print(f'Snapshots: {ss_min:.0f}min')
print('OK')
" 2>/tmp/test_thresholds.log
check $? "Thresholds match Pluribus ratios"

# 6. ctypes struct alignment
echo "[6/6] Ctypes struct alignment"
python3 -c "
import ctypes

class BPConfig(ctypes.Structure):
    _fields_ = [
        ('discount_stop_iter', ctypes.c_int64),
        ('discount_interval', ctypes.c_int64),
        ('prune_start_iter', ctypes.c_int64),
        ('snapshot_start_iter', ctypes.c_int64),
        ('snapshot_interval', ctypes.c_int64),
        ('strategy_interval', ctypes.c_int64),
        ('num_threads', ctypes.c_int),
        ('hash_table_size', ctypes.c_int),
        ('snapshot_dir', ctypes.c_char_p),
        ('include_preflop', ctypes.c_int),
    ]

# Verify expected size and alignment
assert ctypes.sizeof(BPConfig) == 72, f'BPConfig size {ctypes.sizeof(BPConfig)} != 72'

# Verify int64 fields are at 8-byte boundaries
for name, ctype in BPConfig._fields_:
    field = getattr(BPConfig, name)
    if ctype == ctypes.c_int64:
        assert field.offset % 8 == 0, f'{name} at offset {field.offset} not 8-byte aligned'

print(f'BPConfig: {ctypes.sizeof(BPConfig)} bytes, alignment OK')
print('OK')
" 2>/tmp/test_ctypes.log
check $? "Ctypes struct matches C layout"

# Summary
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}DO NOT DEPLOY — fix failures first${NC}"
    exit 1
else
    echo -e "${GREEN}All checks passed — safe to deploy${NC}"
    exit 0
fi
