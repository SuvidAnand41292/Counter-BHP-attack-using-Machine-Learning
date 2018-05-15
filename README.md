# Counter-BHP-attack-using-Machine-Learning

Download the data set from UCI repository using the link:

https://archive.ics.uci.edu/ml/datasets/Burst+Header+Packet+%28BHP%29+flooding+attack+on+Optical+Burst+Switching+%28OBS%29+Network#

The page will have Data Set Discription and Data Folder. Download from the DataFolder Link and save it to your root directory.

Run the 259_final.ipynb file on JUPYTER platform which is availble on ANACONDA software.

You will be able to run the code as mentioned in the comment along witht the code.

Discussion-The Dataset was catalogued as the training and test dataset. Next, the training data set was pre-processed to remove noise or statistical biases. Once the data was cleaned, a filtering process took place to detect the most influential features, as well as reduce the dimensionality of the data. After selecting the features, different classification methods like Logistic, LDA/QDA, and Random Forest algorithm processed the dataset to build a classification model that will then be transformed into rule sets. Then, we checked the performance of the test data set against the rule set framed.

CONCLUSION-We can conclude that complex model is performing well. This is the case because the dataset may be non-linear due to which the complex method like Random Forest are performing well. Even Polynomial Classifier fail.The results show that we can counter the flooding attack on the OBS network if train the nodes using TREE CLASSIFIERS such as RANDOM FOREST which perfectly classifies the nodes to one of the attributes depending on the feature description.
