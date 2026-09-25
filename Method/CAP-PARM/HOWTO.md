# CAP-PARM

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP-PARM/scripts/reproduce_fixed_parm.py \
    --num-prompts 1000 \
    --w 0.5 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --output-path results/cap_parm.json
```

Kết quả: `results/cap_parm.json`

Không cần train.
