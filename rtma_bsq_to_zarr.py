import rioxarray
import xarray as xr
import os
import glob
import warnings
import time
import logging
import datetime

logging.basicConfig(format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    filename=f'rtma_to_zarr_{datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d_%H:%M:%S")}.log')

def sanitize_attrs(attrs):
    """
    Recursively converts numpy number types in a dictionary or list to standard Python types.
    This is crucial for avoiding serialization issues with some backends or libraries
    that might misinterpret numpy types.
    """
    if isinstance(attrs, dict):
        return {k: sanitize_attrs(v) for k, v in attrs.items()}
    elif isinstance(attrs, (list, tuple)):
        return [sanitize_attrs(v) for v in attrs]
    elif hasattr(attrs, 'item'):  # Catches numpy numbers (int, float, etc.)
        return attrs.item()
    else:
        return attrs

def subdir_to_xr(data_dir:str,variable_prefix:str)-> xr.DataArray:
    file_pattern = f"{variable_prefix}*.bsq"
    search_path = os.path.join(data_dir,file_pattern) # os.path.join(data_dir, "*", file_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        logging.info(f"No files found for pattern: {search_path}")
        return None, None

    logging.info(f"Found {len(files)} files for variable {variable_prefix} to process.")

    start = time.time()
    # Open all files lazily. If a file is corrupt, it's skipped.
    data_arrays = []
    for f in files:
        try:
            # rioxarray.open_rasterio with chunks=True will return a dask array
            da = rioxarray.open_rasterio(f, chunks={'band':8760,'x':40,'y':40})
            da = da.isel(band=slice(0,8760))
            data_arrays.append(da) # rioxarray.open_rasterio(f, chunks=(8760,40,40))
        except Exception as e:
            warnings.warn(f"Could not open file {f}. Error: {e}. Skipping.")

    if not data_arrays:
        logging.info(f"Could not open any valid files for variable '{variable_prefix}'.")
        return None

    # Concatenate all DataArrays along a new 'file' dimension.
    # This will create one large dask-backed DataArray.
    combined_data = xr.concat(data_arrays, dim='band')

    end = time.time()
    logging.info(f"Data Read and Concat time: {(end-start)/60} mins")
    logging.info(f"Variable {variable_prefix}:")
    logging.info(f"  Combined data shape:{combined_data.shape}")
    logging.info(f"  Dask chunks {combined_data.chunks}")

    return combined_data

if __name__ == '__main__':
    parent_dir = "datasets/rtma/"
    
    # The project seems to be interested in 'ws' (wind speed) and 'wd' (wind direction).
    
    for var_prefix in ["ws", "wd"]:
        for i,tile in enumerate(sorted(os.listdir(parent_dir))):
            if i>1:
                break
            tile_dir = os.path.join(parent_dir, tile)
            logging.info(tile_dir)
            xd = subdir_to_xr(tile_dir, var_prefix)
            if xd is None:
                logging.info(f"Skipping {tile}, no readable files")
            else:
                # # Sanitize attributes to prevent downstream serialization/deserialization errors.
                # # This converts numpy-specific types (like np.float64) to standard Python types.
                # don't think this solved anything, after fresh py virtenv was able to load zarr from gcs fine
                # logging.info("Sanitizing metadata attributes...")
                # xd.attrs = sanitize_attrs(xd.attrs)
                # for coord in xd.coords:
                #     xd.coords[coord].attrs = sanitize_attrs(xd.coords[coord].attrs)

                try:
                    logging.info("Exporting to zarr...")
                    xd.to_zarr(store=f"gs://delos-downscale/data/rtma/zarr/{tile}_{var_prefix}",mode="w-")
                except FileExistsError as e:
                    logging.info(f"File already exists. {e}")
                except Exception as e:
                    logging.info(f"Can't write file. {e}")