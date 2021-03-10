
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
    def decision_tree_learning(examples, attributes, parent_examples, classification_index=0):
        if len(examples) == 0: return Tree.plurality_value(parent_examples) #return the most common output
        if Tree.same_classification(examples): return examples[0][classification_index] #return the class of the first element, since every is the same
        if len(attributes) == 0: return Tree.plurality_value(examples) #if attribute is empty return most common output



    '''
    @breif return the most common output from the dataset

    @param data the dataset
    @param classification_index the index in the dataset where the class is given

    @return str the most common classification
    '''
    @staticmethod
    def plurality_value(data, classification_index=0):
        count = {}
        for line in data:
            value = line[classification_index] #the classification in the dataset

            if value in count:
                count[value] += 1 #add one more count
            else:
                count[value] = 1
        
        return max(count, key=count.get) #get the max key

    '''
    @breif Check if all the data in a dataset have the same classification
    
    @param data the dataset
    @param classification_index the index in the dataset where the class is given

    @return bool returns true if the classification is same
    '''
    @staticmethod
    def same_classification(data, classification_index=0):
        first_class = data[0][classification_index] #the first class

        for line in data:
            classification = line[classification_index] #Classification from dataset

            if classification != first_class: #if a class is different, they are not the same
                return False
        
        return True
