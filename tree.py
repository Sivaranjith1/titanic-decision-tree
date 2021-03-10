
class Tree:
    def __init__(self):
        pass

    def __str__(self):
        return "Hello tree"

    '''
    @brief Train a tree on example data provided to the function

    @param examples the training data
    @param attributes
    @param parent_examples the training data of the parent

    @return tree a fully trained tree 
    '''
    @staticmethod
    def decision_tree_learning(examples, attributes, parent_examples):
        if len(examples): return Tree.plurality_value(parent_examples) #return the most common output

    '''
    @breif return the most common output from the dataset

    @param data the dataset
    @param classification_index the index in the dataset where the class is given

    @return int the most common classification
    '''
    @staticmethod
    def plurality_value(data, classification_index=0):
        count = []
        for line in data:
            value = int(line[classification_index]) #the classification in the dataset
            while len(count) <= value:
                count.append(0) #adds 0 cases at this index
            
            count[value] += 1 #add one more count
        
        return count.index(max(count))
