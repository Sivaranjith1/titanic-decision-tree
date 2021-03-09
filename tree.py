
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
        if len(examples): return none #need to do
