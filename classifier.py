import sys
import csv
import math
import time
import scipy.stats

class Node:
    def __init__(self, data):
        self.data = data
        self.examples = []
        self.children = {}

    def add_child(self, value, obj):
        self.children[value].append(obj)

def decision_tree_learning(examples,attributes, parent_examples, truth_value):
    # If examples is empty then return plurality value of parent_examples
    if not examples:
        return find_mode(parent_examples)

    # if all examples have the same classification, return the classification
    elif isSame(examples, attributes):
        #get the first values of the answer col
        return examples[0][0]

    # If attributes is empty, return plurality value examples
    elif not attributes:
        return find_mode(examples)
    else:
        # Calculate the attribute to be used as root, and also the index
        best_attribute = dict()
        # Set max_importance to a low value
        max_importance = -1
        for attribute in attributes:
            if find_index(attribute) != 0:
                importance = calculate_importance(attributes[attribute],examples,find_index(attribute),truth_value)
                if importance > max_importance:
                    max_importance = importance
                    best_attribute = attribute

        index = find_index(best_attribute)

        # Remove the best attribute from the dictionary
        attributes_without_best = attributes.copy()
        attributes_without_best.pop(best_attribute, None)

        # Assign the best attribute to the root and create a new decision tree with root test of best attribute
        tree = Node(best_attribute)
        tree.examples = examples

        for value in attributes[best_attribute]:
            updated_examples = []
            for example in examples:
                if value == example[index]:
                    updated_examples.append(example)

            # Recursive call
            tree.children[value] = decision_tree_learning(updated_examples,attributes_without_best,examples,truth_value)
        return tree

def decision_tree_testing(tree, examples, truth_value):
    global attribute_names

    # variables to keep track of metrics
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    # For each example in the test set
    for example in range (0, len(examples)):
        # While there are still nodes to traverse
        temp_tree = tree
        while type(temp_tree) != str:
            # Get the index for the attribute we are splitting on, then go down that branch on the tree according to answer
            index = find_index(temp_tree.data)
            temp_tree = temp_tree.children[examples[example][index]]

        # Record Metrics
        result = temp_tree
        if result == truth_value:
            if result == examples[example][0]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if result == examples[example][0]:
                true_negative += 1
            else:
                false_negative += 1

    # Print output
    print("After testing: ")
    print("True Positive: %d " % true_positive)
    print("False Positive: %d " % false_positive)
    print("True Negative: %d " % true_negative)
    print("False Negative: %d " % false_negative)

# Gather Tree Metrics using a Breadth first approach:
def gather_metrics(tree):
    # Establish nodes to keep track of information
    total_nodes = 1
    decision_nodes = 0
    max_depth = 0
    min_depth = float('inf')
    depth = 1
    leaf_depth = []
    avg_count = 0
    queue = [tree]

    print("\nTree Metrics:")
    # While there are still trees to analyze
    while queue != []:
        tree = queue[0]
        queue.pop(0)
        if type(tree) != str:
            # At each level of the tree, add all the children and evaluate them
            for child in tree.children:
                queue.append(tree.children[child])
                if type(tree.children[child]) == str:
                    total_nodes += 1
                    if depth < min_depth:
                        min_depth = depth
                    if depth > max_depth:
                        max_depth = depth
                    leaf_depth.append(depth)
                    decision_nodes += 1
                else:
                    total_nodes += 1
            depth += 1

    # Print Statistics
    print("Total Nodes: %d" % total_nodes)
    print("Decision Nodes: %d" % decision_nodes)
    print("Maximum Depth: %d" % max_depth)
    print("Minimum Depth: %d" % min_depth)
    print("Average Depth of Root-to-Leaf: %f" %  (sum(leaf_depth)/len(leaf_depth)))

# Function to print the tree
def print_tree(tree, block):
    if type(tree) != str:
        print(block + "Testing " + tree.data)
        for child in tree.children:
            print("   "+block + "Branch " + child)
            print_tree(tree.children[child], block+"        ")
    else:
        print(block[5:]  + "Leaf with value " + tree)
    return

# Find index - Takes the best attribute and return the index of it.
def find_index(attribute):
    global attribute_names
    count = 0
    for item in attribute_names:
        if item == attribute:
            break
        count = count + 1

    return count

# Returns true if all examples have the same classification
def isSame(examples, attributes):
    lst = []
    for x in examples:
        # Append the answers in the table
        lst.append(x[0])
    return all(ele == lst[0] for ele in lst)

# Returns most frequent value
def find_mode(examples):
    lst = []
    for x in examples:
        # Append the answers in the table
        lst.append(x[0])
    return max(set(lst), key = lst.count)

# Read examples from csv, returns 2D list
def read_csv(filename):
    global attribute_names
    examples = []

    with open(filename,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            examples.append(line)
    attribute_names = examples.pop(0)
    return examples

#Get attributes and values in a dictionary
def read_attributes(attributes_file):
    attributes = dict()
    with open(attributes_file) as fp:
        for line in fp:
            current_line = line
            divided = current_line.split(",")
            key = divided[0]
            values = []
            for x in range(1,len(divided)):
                values.append(divided[x].strip())
            attributes[key] = values
    return attributes

# Calculate the importance of the attributes
def calculate_importance(attribute,examples,index,truth_value):
    N = len(examples)
    count = 0
    for example in examples:
        if example[0]==truth_value:
            count+=1
    gain = calculate_entropy(count/N)

    remainder = 0
    for value in attribute:
        value_count=0
        truth_count=0
        subset = []
        for example in examples:
            if example[index] == value:
                subset.append(example)
                value_count+=1
        for p in subset:
            if p[0]==truth_value:
                truth_count+=1
        if N != 0 and value_count != 0:
            remainder += (value_count/N) * calculate_entropy(truth_count/value_count)
    return gain - remainder

# Calculate the entropy
def calculate_entropy(p):
    if (p==0 or p==1):
        return 0.0
    else:
        return -1*(p*math.log2(p) + (1-p)*math.log2(1-p))

# Chi Pruining
def chi_pruning(tree, truth_value, significance):
    prunable = True
    for val in tree.children:
        if type(tree.children[val]) != str:
            tree.children[val] = chi_pruning(tree.children[val], truth_value, significance)
        if type(tree.children[val]) != str:
                prunable = False
    if prunable:
        check_prune = pruning_calculations(tree, tree.examples, significance, truth_value)
        if check_prune:
            mode = find_mode(tree.examples)
            return mode
    return tree

# Function to do the pruning calculation per attribute
def pruning_calculations(node,examples,significance, truth_value):
    should_prune = False
    #Get number of positive and negative examples
    p=0
    n=0
    for example in examples:
        if example[0]==truth_value:
            p+=1
    n = len(examples)-p

    pk =[]
    nk= []
    pHat = []
    nHat = []
    pcount =0
    ncount=0
    index=0
    delta=0

    # Find pK for each attribute
    for branch in node.children:
        pcount=0
        ncount=0
        for example in examples:
            if example[find_index(node.data)] == branch and example[0]==truth_value:
                pcount+=1
            elif example[find_index(node.data)] == branch and example[0]!=truth_value:
                ncount+=1

        pk.append(pcount)
        nk.append(ncount)
        index+=1

    # Calculate pHat and nHat values
    for i in range(len(pk)):
        pHat.append(calculateHat(p,pk[i]+nk[i],p+n))
        nHat.append(calculateHat(n,pk[i]+nk[i],p+n))
    val = scipy.stats.chi2.ppf(1-significance,len(pHat)-1)

    # Calculate delta
    for i in range(len(pHat)):
        delta += calculate_delta(pk[i],pHat[i],nk[i],nHat[i])

    if delta < val:
        should_prune = True
    return should_prune

def calculateHat(constant,top,bottom):
    return constant * (top/bottom)

def calculate_delta(pk, pkHat, nk, nkHat):
    a=0
    b=0
    if pkHat != 0: # If node already pruned, risked the chance of divide by 0  because mode could give slit to only positive or negative
        a = pow(pk-pkHat,2)/pkHat
    if nkHat != 0:
        b = pow(nk-nkHat,2)/nkHat
    return a + b


attribute_names = []
def main():
    """
    if len(sys.argv) > 4:
        print("Usage: python3 classifier.py <attributes> <training-set> <testing-set> <significance>")
    """
    global attribute_names

    attributes_file = sys.argv[1]
    training_set= sys.argv[2]
    testing_set= sys.argv[3]
    significance=float(sys.argv[4])

    attributes = read_attributes(attributes_file)
    train_examples = read_csv(training_set)
    test_examples = read_csv(testing_set)

    #The value we use as true value
    truth_value = train_examples[0][0]

    # Build decision tree
    print("Building the decision tree with %s" % training_set)
    root = decision_tree_learning(train_examples,attributes,train_examples, truth_value)

    # Print Tree
    print("\nPrinting Tree BEFORE pruning:")
    print_tree(root, "")
    gather_metrics(root)

    # Test the data
    print("\nTesting the Tree with testing set %s BEFORE pruning" % testing_set)
    decision_tree_testing(root, test_examples, truth_value)

    # Prune the tree
    pruned_tree = chi_pruning(root, truth_value, significance)

    # Print Tree
    print("\nPrinting Tree AFTER pruning:")
    print_tree(pruned_tree, "")
    gather_metrics(pruned_tree)

    # Test the data
    print("\nTesting the Tree with testing set %s AFTER pruning" % testing_set)
    decision_tree_testing(pruned_tree, test_examples, truth_value)

if __name__ == "__main__":
    main()
