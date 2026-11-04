import h5py
import numpy as np


class loadData():
	def stream_category(file_path, category_name):
		with h5py.File(file_path, "r") as f:
			category_group = f[category_name]
			
			for asin, asin_group in category_group.items():
				matrices = [dataset[:] for dataset in asin_group.values()]
				yield asin, matrices

def main():
	count = 0
	totalReviewCount = 0
	for asin, matrices in loadData.stream_category("arrays.h5", "Electronics"):
		print(f"product: {asin}")
		print(f"Total Reviews for item: {len(matrices)}")
		print()
		count += 1
		if count > 100: break
		#totalReviewCount += len(matrices)
	print(totalReviewCount)



if __name__ == "__main__":
    main()