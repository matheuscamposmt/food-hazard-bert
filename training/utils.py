
import json
def pretty_print_report(epoch, metrics):
    print(f"\n[REPORT] Validation Metrics for Epoch {epoch}:\n")
    if "st1" in metrics:
        print("\nMacro average")
        print(json.dumps(metrics["st1"]["macro avg"], indent=4))

    if "st2" in metrics and metrics["st2"] is not None:
        print("\nMacro average")
        print(json.dumps(metrics["st2"]["macro avg"], indent=4))
