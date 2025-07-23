# JGS_llama31_UT


## How To Run

```bash
# Save output tensors on PVC
UT_MODE=DUMP pytest -sv test_op_4-func.py

# Compare output tensors on JGS simulator
UT_MODE=COMPARE pytest -sv test_op_4-func.py
```