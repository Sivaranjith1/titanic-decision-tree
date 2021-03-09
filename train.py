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


if __name__ == '__main__':
    trainingData = load_csv('dataset/train.csv')
    
    t1 = tree.Tree()

    print(trainingData[0])