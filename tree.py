import math
from graphviz import Digraph

class Tree:
    '''
    @breif creates a decision tree

    @param attribute the attribute this has tree has rot in
    '''
    def __init__(self, attribute):
        self.attribute = attribute

        self.branch = {} #branch with key equal attribute_name and value is tree

    def __str__(self):
        string = f"{self.attribute} \n"

        for branch in self.branch:
            string += str(branch) + "\t"
            string += f"{self.branch[branch]} \t"
            string += "\n"

        return string

    '''
    @breif add a branch with attribute name as subtree

    @param attribute_name The value to add as key
    @param tree the tree to add as subtree
    '''
    def add_branch(self, attribute_name, tree):
        self.branch[attribute_name] = tree

    '''
    @breif predict the output of the data
    
    @param data an dict with attributes

    @return the predicted case
    '''
    def predict(self, data):
        attribute_value = data[self.attribute] #the value in the dataset for this attribute
        subtree = None
        if attribute_value in self.branch.keys():
            subtree = self.branch[attribute_value] #get the correct subtree for this attribute value
        else:
            subtree = 0
        if isinstance(subtree, Tree): #if subtree is a tree
            return subtree.predict(data) #go through the subtree
        else: #if subtree is a value
            return subtree #return the output

    def start_dot(self):
        dot = Digraph()
        self.add_to_dot(dot, None, 0)
        dot.render('test-output/round-table.gv', view=True)

    def add_to_dot(self, dot, parent_name, index):
        dot.node(f"{self.attribute}{index}", f"{self.attribute}")
        if parent_name != None:
            dot.edge(parent_name, f"{self.attribute}{index}")

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

        max_attribute = Tree.importance(attributes, examples) #attribute witht he highest importance gain
        current_tree = Tree(max_attribute) #create a new tree with rot in max attribute

        attributes_names = Tree.get_values_of_attribute(max_attribute, examples) #get the different values in attribute
        
        for vk in attributes_names:
            exs = Tree.get_data_from_attribute(max_attribute, vk, examples) #gets the data examples with attribute name vk

            new_attribute = attributes.copy()
            del new_attribute[attributes.index(max_attribute)] #remove max attribute from the copy of attribute

            subtree = Tree.decision_tree_learning(exs, new_attribute, examples) #training subtree
            current_tree.add_branch(vk, subtree) #add subtree to current tree
        return current_tree

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


    '''
    @breif gets all the attribute names of a given attribute

    @param attribute the key in dataset
    @param data the dataset to go through

    @return string[] return a array of values
    '''
    @staticmethod
    def get_values_of_attribute(attribute, data):
        values = []
        for line in data:
            attr = line[attribute]
            if attr not in values:
                values.append(attr)

        return values

    '''
    @breif get the data where attribute has the value attribute_name

    @param attribute the attribute to look at
    @param attribute_name the value to check for
    @param data the dataset

    @return array the new dataset
    '''
    @staticmethod
    def get_data_from_attribute(attribute, attribute_name: str, data):
        exs = [] #output where attribute has the value attribute_name

        for line in data:
            if line[attribute] == attribute_name:
                exs.append(line)

        return exs