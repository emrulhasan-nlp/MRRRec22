Hello!
This is the implementation of our paper MRRRec2022 IEEE big data

##################################################################################################################################################
# Please note that our paper is the extension of ANR: Aspect-based Neural Recommender" in CIKM 2018. Therefore, we added code to their implementation (https://github.com/almightyGOSU/ANR/tree/master). The architecture that we extended, we DEVELOP THE CODE FOR THAT PART ONLY. The detailed architecture of our model can be found in the paper (https://ieeexplore.ieee.org/abstract/document/10020741). Thank you very much to the author of ANR for their nice work and open-source work!

#######################################################################################################################


# Implementation of our paper "MRRRec: Multi-criteria Rating and Review based Recommendation Model" in IEEE bigdata 2022.

The paper is available here: [Paper](https://ieeexplore.ieee.org/abstract/document/10020741)



## Environment Setup

1. Python 3.11.5
2. PyTorch 2.1.2+cud121

To run the code, please follow the steps below.

Step 0. We assume the following directory structure:

NAME your root directory                               # root folder

GoogleNews-vectors-negative300.bin    # This is the file for pretrained word embeddings

MRRRec/                                  # Basically, what you clone from github..

    __saved_models__/                   # This is where the pretrained MRRRec weights go to..

    datasets/                           # Place the tripadvisor csv file here
      clean_tripdata/             

    experimental_results/               # This is where your results go to..

    model/                              # All model-related code (i.e. all the PyTorch stuff)

    preprocessing/                      # All preprocessing code 

    FILEPATHS.py                        # Names of files shared across all code
    PyTorchTEST.py                      # Basically main.py.. The model is trained and tested here (despite the weird filename)



Step 1. Preprocessing

  - NOTE: For this step, your current directory should be the 'preprocessing' folder..
  - E.g. MRRRec/preprocessing/ in the example directory structure!
	run the following two commands
	
	a. python preprocessing_simple.py -d clean_tripdata -dev_test_in_train 1
	b. python pretrained_vectors_simple.py -d clean_tripdata


Step 2. Running the model
	Come back to the folder MRRRec and run the following command

    python PyTorchTEST.py -d "clean_tripdata" -m "MRRRec" -e 25 -p 1 -rs 1337 -gpu 0 -vb 1

  - model output (some information & results) are saved to the 'experimental_results' folder
  - e.g. MRRRec/experimental_results/clean_tripdata - MRRRec/2022-12-05-21-28-46-logs.txt


##########Tripadvisor dataset##############

dataset is available here: https://www.cs.virginia.edu/~hw5x/Data/LARA/TripAdvisor/

Follow the instructions in the paper (https://ieeexplore.ieee.org/abstract/document/10020741) to clean the dataset.



## Please consider citing our work if you find it useful. Thanks!

@inproceedings{hasan2022multi,
  title={Multi-criteria Rating and Review based Recommendation Model},
  author={Hasan, Emrul and Ding, Chen and Cuzzocrea, Alfredo},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={5494--5503},
  year={2022},
  organization={IEEE}
}



