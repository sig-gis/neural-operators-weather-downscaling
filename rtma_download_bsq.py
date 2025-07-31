import os
import subprocess
import shutil
from pathlib import Path
import glob

parent = "datasets/rtma"
Path(parent).mkdir(parents=True, exist_ok=True)

# print("Copying rtma data from GCS...")
# prc = subprocess.run(f"gcloud storage rsync gs://delos-downscale/data/rtma {parent} --recursive --no-clobber",
#                            capture_output=True,
#                            text=True,
#                            shell=True)
# if prc.returncode != 0:
#     raise RuntimeError(prc.stderr)

folders = sorted(os.listdir(parent))

for i,folder in enumerate(folders):
   print(f"processing tile folder {folder}...")
   
   archives = sorted(glob.glob(f"{parent}/{folder}/*.tgz"))
   print(f"{len(archives)} archives found")
   if len(archives)>0:
      for archive in archives:
         archive = Path(archive)
         print(f"Extracting {archive} to {archive.parent}")
         shutil.unpack_archive(archive, archive.parent)
   # in addition to removing the archive file, we discard the 2023 files because they are incomplete
   to_remove = [os.path.join(parent, folder, file) for file in sorted(os.listdir(f"{parent}/{folder}")) if file.endswith(".tgz") or "2023" in file]
   for file in to_remove:
      print(f"removing {file}")
      os.remove(file)

         
   