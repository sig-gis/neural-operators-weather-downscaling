import numpy as np
import json
def pooled_sd(vars):
    """square root of mean variance
        
        """
    pooled_var = np.mean(vars)
    pooled_sd = np.sqrt(pooled_var)
    return pooled_sd

for var in ["ws","wd"]:
    json_file = f"rtma_{var}_stats.json"
    stats = json.load(open(json_file))
    print(stats)
    means,stds,vars = stats["means"],stats["stds"],stats["vars"]

    pooled_mean = np.mean(means)
    pooled_std = pooled_sd(vars)
    print(pooled_mean)
    print(pooled_std)
    print()


