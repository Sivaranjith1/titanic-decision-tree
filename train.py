from tree import Tree
from csv import reader

'''
    @breif Returns the rows in a csv as a array

    @param filename the csv file to read from

    @return an 2d array of csv data, where each element of the parent array is a line
'''
def load_csv(filename):
    data = []
    open_file = open(filename, 'r')
    for line in reader(open_file):
        data.append(line)
    open_file.close()

    return data[1:]

'''
@breif finds the missing columns in the dataset, missing column is defined as when the len of data is zero

@param data the dataset

@return missing_columns An reverse sorted array with unique indexs of missing columns
'''
def find_missing_columns(data):
    missing_columns = [] #clear the missing column storage
    for line in data:
        for index, data_point in enumerate(line):
            if(len(data_point) == 0): #the column is missing
                missing_columns.append(index)

    return sorted(set(missing_columns), reverse=True)

'''
@breif Removes the columns from the dataset

@param data the dataset to delete from
@param columns an reverse sorted list of columns to be deleted

@return data the new dataset where columns are removed
'''
def remove_columns(data, columns):
    edited_data = []
    for line in data:
        for column in columns:
            del line[column]
        edited_data.append(line)

    return edited_data

'''
@breif change the lines in the dataset to be dicts

@param data the dataset

@retun array of dict returns the dataset where the lines have been replaced with dicts
'''
def convert_line_to_dict(data):
    edited_data = []
    for line in data:
        new_line_dict = {}
        for i, column in enumerate(line):
            new_line_dict[i] = column
        edited_data.append(new_line_dict)
    return edited_data

def get_accuracy(trained_model, dataset):
    correct_guess = 0
    for line in dataset:
        if trained_model.predict(line) == line[0]:
            correct_guess += 1

    return correct_guess/len(dataset)

if __name__ == '__main__':
    #load the datasets
    training_data   = load_csv('dataset/train.csv')
    test_data       = load_csv('dataset/test.csv')

    #find missing columns
    missing_columns = find_missing_columns(training_data)

    #remove these missing columns from both datasets
    training_data   = remove_columns(training_data, missing_columns)
    test_data       = remove_columns(test_data, missing_columns)

    #convert both datasets to dicts
    training_data   = convert_line_to_dict(training_data)
    test_data       = convert_line_to_dict(test_data)

    print(training_data[0])

    trained_w_categorical = Tree.decision_tree_learning(training_data, [1, 3, 8], []) #the trained decision tree for only categorical values


    print(f"Accuracy on the testdata on model with only categorical value: \n\t {get_accuracy(trained_w_categorical, test_data)}")