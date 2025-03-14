import os
import utils
import gen_data
import train_models
import evaluate_model

if __name__ == "__main__":
	print("starting the ecg cnn pipeline")

	print("\n \n \n generating data")
	gen_data.main()

	print("\n \n \n training models")
	train_models.main()

	print("\n \n \n evaluating models")
	evaluate_model.main()

	print("\n \n \n pipeline finished, find results in parent directory")


