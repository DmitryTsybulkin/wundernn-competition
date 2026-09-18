# Vanilla GRU ONNX baseline

This is an inference-only stateful baseline. It scores **0.589595 WP** on the
complete validation set.

From the starterpack root, run:

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The model updates its recurrent state on every row, including warm-up rows,
and resets state between sequences. ONNX Runtime is configured for CPU-only
sequential execution with one intra-op and one inter-op thread.

`baseline_submission.zip` contains `solution.py` and `baseline.onnx` at the
archive root and is ready to upload. The baseline does not include training
code or a training checkpoint.


## Result of the baseline run
I tried to run the baseline solution on my local machine and got the following output:

```
main@skynet-378 wnn_connectome_starterpack % python3 baseline/solution.py --validation datasets/valid.parquet
{
	't0': 0.5714917100207924, 
	't1': 0.6076973122694173, 
	'weighted_pearson': 0.5895945111451049, 
	'blocks': 1873, 
	'eligible_blocks': 1872, 
	'excluded_blocks': 1, 
	'selected_rows': 3528273, 
	'effective_scored_rows': 3527577, 
	'metric_clip': [-2.0, 2.0], 
	'aggregation': 'equal mean of eligible block-wise two-target clipped Weighted Pearson',
	'rows_seen': 37460000, 
	'predictions_required': 37274573
}
```