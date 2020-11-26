# BlueNose
A classification model to predict molecule odor

### Data
train.csv: Training data. Structure: molecule,scent_0,scent_1,etc...

test.csv: Testing data. Structure: molecule\n

vocab.txt: txt file where each word contains a possible scent.

### Preprocessing
To preprocess the data, you should call the function 'get_data', which takes in train.csv, test.csv, and vocab.txt. 

The training labels are in a one hot vector style, where each label is represented as a vector. Each of this label's 
classes are first converted to integers, then a 1 is placed at the index of each integer. 
The difference between these vectors and one hot vectors is a label vector may contain multiple classes, 
and therfore multiple 1's. 
