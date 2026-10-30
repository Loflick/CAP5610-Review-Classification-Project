import h5py
import numpy as np

def stream_category(file_path, category_name):
	with h5py.File(file_path, "r") as f:
		category_group = f[category_name]
        
		for asin, asin_group in category_group.items():
			matrices = [dataset[:] for dataset in asin_group.values()]
			yield asin, matrices

for asin, matrices in stream_category("arrays.h5", "Electronics"):
	print(f"product: {asin}")
	print(f"Total Reviews for item: {len(matrices)}")