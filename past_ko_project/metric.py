import torch
import sklearn.metrics as metrics

def compute_metric(metric, output, target, average="binary"):
    if metric == "mse":
        return metrics.mean_squared_error(target.cpu(), output.cpu())
    elif metric == "acc":
        preds = torch.argmax(output.cpu(), dim=1)
        return metrics.accuracy_score(target.cpu(), preds)
    elif metric == "f1score":
        preds = torch.argmax(output.cpu(), dim=1)
        return metrics.f1_score(target.cpu(), preds, average=average)
    else:
        raise NotImplementedError(f'Metric {metric} is not supported.')
