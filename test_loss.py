import mlflow
mlflow.login()
mlflow_client =mlflow.tracking.MlflowClient()

metrics = [
    "loss/train/total",
    "lr-DecoupledAdamW/group0",
    "memory/alloc_retries",
    "memory/current_active_mem",
    "memory/current_allocated_mem",
    "memory/current_inactive_mem",
    "memory/current_reserved_mem",
    "memory/peak_active_mem",
    "memory/peak_allocated_mem",
    "memory/peak_inactive_mem",
    "memory/peak_reserved_mem",
    "metrics/eval/LanguageCrossEntropy",
    "metrics/eval/LanguagePerplexity",
    "metrics/eval/TokenAccuracy",
    "metrics/train/LanguageCrossEntropy",
    "metrics/train/LanguagePerplexity",
    "metrics/train/TokenAccuracy",
    "throughput/batches_per_sec",
    "throughput/device/batches_per_sec",
    "throughput/device/flops_per_sec",
    "throughput/device/mfu",
    "throughput/device/samples_per_sec",
    "throughput/device/tokens_per_sec",
    "throughput/flops_per_sec",
    "throughput/samples_per_sec",
    "throughput/tokens_per_sec",
    "time/batch",
    "time/batch_in_epoch",
    "time/epoch",
    "time/remaining_estimate",
    "time/sample",
    "time/sample_in_epoch",
    "time/token",
    "time/token_in_epoch",
    "time/total",
    "time/train",
    "time/val"
]

for metric_name in metrics:    
    branch_mlflow_history = mlflow_client.get_metric_history('07e2015fd32e44f4bf0eab72ac21b816', metric_name) #https://dbc-04ac0685-8857.staging.cloud.databricks.com/ml/experiments/2044406646765817/runs/07e2015fd32e44f4bf0eab72ac21b816
    dev_mlflow_history = mlflow_client.get_metric_history('5dcfb027aa8644978da26df63adc247f', metric_name) #https://dbc-04ac0685-8857.staging.cloud.databricks.com/ml/experiments/2044406646765817/runs/5dcfb027aa8644978da26df63adc247f/model-metrics?o=3360802220363900
    avg_branch = 0
    avg_dev = 0
    for branch_metric, dev_metric in zip(branch_mlflow_history, dev_mlflow_history):
        avg_branch += branch_metric.value
        avg_dev += dev_metric.value
    avg_branch /= len(branch_mlflow_history)
    avg_dev /= len(dev_mlflow_history)
    if avg_branch > avg_dev:
        print(metric_name + ": Non Powers of 2")
    elif avg_branch < avg_dev:
        print(metric_name + ": Powers of 2")
    else:
        print(metric_name + ": Equal")