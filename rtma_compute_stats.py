import xarray as xr
from dask.distributed import Client, LocalCluster
import dask
import time
import json
import subprocess




def compute_stats_for_variable(zarr_store):
    """
    Computes the global mean and standard deviation for a specific variable
    from a dataset of .bsq files in parallel.

    Args:
        data_dir (str): The root directory of the dataset.
        variable_prefix (str): The prefix for the variable files (e.g., "ws").

    Returns:
        tuple: (mean, std) for the variable. Returns (None, None) if no files found.
    """
    da = xr.open_zarr(zarr_store)
    print(da)

    start = time.time()
    print("  Computing global mean and std... (this may take a while)")
    mean,std,var = [float(result.to_dataarray().to_numpy()[1]) for result in [da.mean().compute(),da.std().compute(),da.var().compute()]]
    end = time.time()
    print(f"  xr.DataArray.compute time: {(end-start)/60} mins")
    return mean,std,var

if __name__ == '__main__':
    # Set up a local Dask cluster for parallel processing.
    # cluster = LocalCluster()
    # client = Client(cluster)
    # print("-" * 50)
    # print(client)
    # print(f"Dask dashboard link: {client.dashboard_link}")
    # print("-" * 50)

    parent_dir = "gs://delos-downscale/data/rtma/zarr"
    cmd = f"gcloud storage ls {parent_dir}"
    out = subprocess.run(cmd,shell=True,stderr=subprocess.PIPE,stdout=subprocess.PIPE).stdout.decode("utf-8")
    zarrs = out.splitlines()
    for var_prefix in ["ws", "wd"]:
        print(f"Computing stats for {var_prefix}...")
        global_stats = {}
        means=[]
        stds=[]
        vars=[]
        zarr_var = [z for z in zarrs if var_prefix in z]
        for i,zs in enumerate(zarr_var):
            if i>0:
                break
            print(f"processing {zs}")
            mean,std,var = compute_stats_for_variable(zs)
            means.append(mean)
            stds.append(std)
            vars.append(var)
            
        
        global_stats["means"] = means
        global_stats["stds"] = stds
        global_stats["vars"] = vars
        print(global_stats)
        
        with open(f"rtma_{var_prefix}_stats.json", mode="w") as f:
            f.write(json.dumps(global_stats))
        break

        
    # print("-" * 50)
    # client.close()
    # cluster.close()
    # print("Dask client and cluster closed.")