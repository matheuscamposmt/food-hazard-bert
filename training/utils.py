
import json
def pretty_print_report(epoch, metrics):
    print(f"\n[REPORT] Validation Metrics for Epoch {epoch}:\n")
    if "st1" in metrics:
        print(json.dumps(metrics["st1"], indent=4))