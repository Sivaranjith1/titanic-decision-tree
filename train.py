import tree
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


if __name__ == '__main__':
    trainingData = load_csv('dataset/train.csv')

    missing_columns = find_missing_columns(trainingData)
    trainingData = remove_columns(trainingData, missing_columns)

    print(trainingData[0])