## Scheduled Unfreeze

```json
{
    "skip_reten": {
        "weight_morm": 40000,
        "loss_improvement": {
            "th0.01-pat1": 20000,
            "th0.005-pat2": 45000
        }
    },
    "skip_reten-skip_oproj": {
        "weight_morm": 55000,
        "loss_improvement": {
            "th0.01-pat1": 20000,
            "th0.005-pat2": 45000
        }
    },
    "skip_reten-skip_oproj-skip_mlp": {
        "weight_morm": 40000,
        "loss_improvement": {
            "th0.01-pat1": 30000,
            "th0.005-pat2": 60000
        }
    }
}
```
