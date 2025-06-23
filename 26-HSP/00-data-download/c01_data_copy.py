from roma import console

import os
import shutil



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
src_dir = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_raw'
path_list = [
	'sub-S0001111189359/ses-1',
	'sub-S0001111190905/ses-1',
	'sub-S0001111190905/ses-2',
]

src_list = [os.path.join(src_dir, p) for p in path_list]

dst_dir = r"E:\test_dir"

# -----------------------------------------------------------------------------
# (2) Copy folder from src to dst
# -----------------------------------------------------------------------------
N = len(src_list)
n_success = 0

console.show_status('Copying files ...')
for i, (src, id_ses) in enumerate(zip(src_list, path_list)):
  console.show_status(f'[{i+1}/{len(src_list)}] Copying {id_ses} ...')
  console.print_progress(i, N)

  # Check if source path exists
  if not os.path.exists(src):
    console.warning(f"Source path does not exist: {src}")
    continue

  dst = os.path.join(dst_dir, id_ses)

  if os.path.exists(dst):
    console.show_status(f"Destination path already exists: `{dst}`")
    continue

  try:
    shutil.copytree(src, dst)
    console.show_status(f"Copied `{src}` to `{dst}`")
    n_success += 1
  except Exception as e:
    console.warning(f"Failed to copy `{src}` to `{dst}`: {e}")

console.show_status(f'Successfully copied {n_success} files.')


