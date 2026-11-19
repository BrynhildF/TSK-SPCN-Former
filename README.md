# TSK-SPCN

Model source, pretrained GWD and single-variable SWaT weights (seed 69), and a test script.

```bash
python -m pip install -r requirements.txt
python test.py --dataset GWD
python test.py --dataset SWaT
```

Without `--input`, tests use synthetic data, not benchmark evaluation. For prepared windows, add `--input windows.npy --output results.npz`: GWD expects `(B,12,10)`, SWaT `(B,1,10)`, with the original preprocessing. 
