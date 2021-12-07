# MLProjectFall21

1. Download the dataset from https://www.kaggle.com/c/siim-covid19-detection
which includes a folder named RSNACOVID which you will need to unzip. Inside this folder are a test folder, train folder, and csv files. We only care about the train_study_level.csv which includes. Kindly make sure that the folder has the dataset is called RSNACOVID and is in the same directory as the code. 


2. In order to preprocess data, run the preprocessing.py file as this is expected to run once to preprocess data and not used again.


3. Run the main code which require a number of input specifications which include:
   1. The experiment (technique) you wish to conduct: baseline model (resnet or densenet), transfer learning (imagenet or chexpert) (resnet or densenet), or self-supervised learning (SimCLR or MoCo)
   2. Several paths required for multiple parts of the code
   3. Batch size for the experiment
   4. Number of epochs
   5. Whether you wish to show a histogram for the data (not shown by default)
   6. Whether you wish to show the transformed image (not shown by default)


4. The main code would run your desired experiments with the help of the other '.py' files. The job for each is as follows:
   1. data.py: includes the class to define a chest x-ray dataset as well functions to show histogram for data classes, show transformed images, calculate class weights, and perform mean and standard deviation calculations
   2. train.py: includes a training class which is the base for the training of the different experiments
   3. test.py: includes a testing class which is the base for testing of the different experiments
   4. baseline.py: includes functions for baseline (resnet) and baseline (densenet)
   5. transfer.py: includes functions for transfer learning (ImageNet or CheXpert) (resnet or densenet)
   6. preprocessing_train_chexpert.py: includes a data set preparation function as well as a preprocessing/training function that is essential for the transfer learning CheXpert-pretrained experiments
   7. SimCLR_pretrain.py: builds SimCLR model and saves the model
   8. SimCLR.py: loads model from SimCLR_pretrain.py to carry on SimCLR experiment
   9. MoCo_pretrain.py: builds MoCo model and saves the model
   10. MoCo.py: loads model from MoCo_pretrain.py to carry on MoCo experiment
   11. GradCam.py: includes function to highlight key regions in the image for a particular prediction
   12. Analysis.py: includes functions essential for model evaluation such as a confusion matrix function and learning curve plotting function