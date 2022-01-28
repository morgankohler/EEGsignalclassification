# EEGsignalclassification

Pet project to classify EEG motor imagery. The dataset used is from https://www.nature.com/articles/sdata2018211 which provides 60,000 mental imagery examples from 13 participants.

Right now I am using the CLA-3state sessions to classify right and left hand movement imagery (i.e. participants are asked to imagine
movement of their left and right hands, the third state being neutral or no imagery). I found this dataset through this helpful list of public EEG datasets here: https://github.com/meagmohit/EEG-Datasets.

As of now it is an attention LSTM with some recurrent connections around the LSTM-attention layers. With no recurrent connections the accuracy achieved ~86% with only 6 LSTM layers and a hidden
dimension of 128. The left-right hand imagery is known as an easier task for motor imagery classification which is why this is what I started with. I will try and get this accuracy
higher before I move to more complex motor-imagery classes. 

Right now it is just classifying the different actions as a proof of concept but it should be classifying against background class and be able to tell when the correct sequence is found 
corresponding to the user thinking of a motor imagery task.
