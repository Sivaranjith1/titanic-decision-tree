import math

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


    '''
    @breif Get the argmax attribute of the importance gain function

    @param attributes list of index in the dataset
    @param data the dataset
    @param classification_index index where the classification is given

    @return the attribute with highest importance function
    '''
    @staticmethod
    def importance(attributes, data, classification_index=0):
        positive = 0 #total positive examples
        negative = 0 #total negative examples

        data_by_attribute = {} #Given the attribute as key the value will be a dict of dict with positive and negative

        for attribute in attributes:
            if attribute not in data_by_attribute:
                data_by_attribute[attribute] = {}

        '''
        data_by_attribute = {
            attribute : { attribute_name: {positive: 10, negative: 10}, ...},
            ...
        }
        '''

        first_class = data[0][classification_index] #used to define what is positive in the rest of the function
        for line in data:
            classification = line[classification_index]
            if first_class == classification:
                positive += 1
            else:
                negative += 1

            for attribute in attributes:
                attribute_name = line[attribute]
                if attribute_name not in data_by_attribute[attribute]:
                    data_by_attribute[attribute][attribute_name] = {
                        "positive": 0,
                        "negative": 0,
                    }

                if first_class == classification:
                    data_by_attribute[attribute][attribute_name]["positive"] += 1 #add positive to attribute with discret value attribute_name
                else:
                    data_by_attribute[attribute][attribute_name]["negative"] += 1 #add negative to attribute with discret value attribute_name
    
                
        argmax_attribute = '' #the attribute with the highest importance gain
        max_attibute = float('-inf')
        #get remainder for all attributes
        for attribute in attributes:
            reminder = 0 #the reminder of attribute in the importance gain function
            for key_attribute, value in data_by_attribute[attribute].items():
                pk = value["positive"]
                nk = value["negative"]
                reminder += (pk + nk)*Tree.entropy_boolean_value(pk/(pk+nk))/(positive + negative)

            #check if this has the highest importance gain
            gain = (Tree.entropy_boolean_value(positive/(positive + negative)) - reminder) 
            if gain > max_attibute:
                max_attibute = gain
                argmax_attribute = attribute

        return argmax_attribute


    '''
    @breif the entropy boolean value function
    
    @param q input for the function

    @return float the entropy value of @p{q}
    '''
    @staticmethod
    def entropy_boolean_value(q):
        if q == 0: return 0
        if q == 1: return 0
        return -(q*math.log2(q) + (1-q)*math.log2(1-q))