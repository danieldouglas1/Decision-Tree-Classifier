Decision Tree Learning
----------------------------------

# Introduction

Python program to implement a classifier to learn the structure of data from a set of training examples. Classifier is evaluated using the [congressional voting records dataset][votes], which lists the votes of members of the US House of Representatives on different issues and their classification as a Democrat or a Republican.

The classifier is in the form of a decision tree created from the training examples. The decision tree will then classify test examples as either democrat or republican and the results of the classification will be reported.  
&chi;<sup>2</sup> pruning is applied to the learned tree to compare the results.

[votes]:https://archive.ics.uci.edu/ml/datasets/congressional+voting+records

# Data

Training and testing sets of the congressional voting records dataset are available in the GitHub repository as CSV files: `house-dataset-train.csv` and `house-dataset-test.csv`. Additionally, there is
an attributes file: `house-dataset-attributes.txt` which lists each attribute on a separate line starting
with the attribute name and then the possible values for the attribute, e.g., the first line of the file is:
`response,republican,democrat`

# Usage

To run the program, simply type :
	python3 classifier.py 'attributes file' 'trainining data' 'testing data' 'significance level'

The program outputs the following.
  1. For the training set and the test set.
     - Recognition rate (% correct).
     - A confusion matrix, which includes the number of occurrences for each combination of assigned class (rows) and actual class (columns). The main diagonal contains counts for correctly assigned examples, all remaining cells correspond to counts for different types of error.

  2. A summary of the learned decision tree.
     - Number of nodes and number of leaf (decision) nodes.
     - A printout of the tree. (Use indentation to show the depth of different nodes.)
     - Maximum, minimum, and average depth of root-to-leaf paths in the decision tree.
     - The printout of your tree and the corresponding summary should look like the following.
```
-- Printing Decision Tree --
Testing Patrons
    Branch Some
    Leaf with value: Yes
    Branch Full
    Testing Hungry
        Branch Yes
        Testing Type
            Branch French
            Leaf with value: Yes
            Branch Thai
            Testing Fri/Sat
                Branch No
                Leaf with value: No
                Branch Yes
                Leaf with value: Yes
            Branch Burger
            Leaf with value: Yes
            Branch Italian
            Leaf with value: No
        Branch No
        Leaf with value: No
    Branch None
    Leaf with value: No
Total Nodes: 12
Decision Nodes: 8
Maximum Depth: 4
Minimum Depth: 1
Average Depth of Root-to-Leaf: 2.625
```
