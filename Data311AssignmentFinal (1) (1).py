# import required libraries to the program
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', 500)  # to avoid truncating the output table, needed because IDE was compressing 
# the table of values, just get rid of if causing issues.
df = pd.read_csv("/Users/evanspencer/Downloads/Bank_Customers.csv")   # loading csv file


def getMax(col):
    # find the maximum value of the given data list
    max = 0
    for row in col:
        if row > max:
            max = row
    return max


def getMin(col):
    # find the minimum value of the given data list
    min = col[0]  # set the first value of the list as the initial minimum value
    for row in col:
        if row < min:
            min = row

    return min


def getAvg(col):
    # calculate the average of the given data list
    sum = 0
    for row in col:  # calculate the sum
        sum += row # adding the rows together

    return sum / len(col)  # divide it by number of elements


def getStd(col):
    #caluculating standard deviation
    mean = getAvg(col)  # get the average or mean value of the data list
    standard_d = 0
    for row in col:  # calculate the summation of squared differences
        standard_d += (row - mean) ** 2

    standard_d = (standard_d / len(
        col)) ** 0.5  # divide squared difference summation by number of data points and take the square root of it
    return standard_d


def getMostCommon(col):
    # find the most common element in the given data list
    values_count = {}
    for row in col:  # store element count in a dictionary
        if row in values_count:
            values_count[row] += 1  # when the key already exists in the dictionary
        else:
            values_count[row] = 1  # when the key doesn't exists in the dictionary

    most_common = list(values_count.keys())[0]  # assign an initial value for the most common value
    temp_val = list(values_count.values())[0]  # assign an initial value for the element count for comparison
    for k in list(values_count.keys()):  # compare the dictionary values and find the most common element
        if values_count[k] > temp_val:
            most_common = k  # update the most commom element by dictionary key

    return most_common, values_count  # return the most common element and the element count dictionary


def numeric_check(col):
    # check whether the given data list has non-integer of non-decimal values
    for row in col:
        if not (isinstance(row, int) or isinstance(row, float)) == True:
            return False

    return True


column_list = list(df.columns)  # create a list of data column headings
data_row_count = df.shape[0]  # store the number of data rows in a variable
outputs = {}
for col in column_list:  # iterate over each data column in the dataframe
    if numeric_check(df[col]) == True:  # when all the elements are numeric, we calculate max,min and average
        outputs[col] = [getMax(df[col]), getMin(df[col]), getAvg(df[col])]
    else:
        outputs[col] = ['NA', 'NA', 'NA']

    if numeric_check(df[col]) == True and len(
            df[col]) >= 2:  # when there is atleast 2 numeric elements in the list, we calculate STD
        outputs[col].append(getStd(df[col]))
    else:
        outputs[col].append('NA')

    outputs[col].append(getMostCommon(df[col])[0])  # find the most common element

data_table = [['Column name', 'MAX', 'MIN', 'AVG', 'Standard deviation', 'Most common value']]
for k in list(outputs.keys()):  # store all the outcomes in a nested list
    data_table.append([k] + outputs[k])

output_df = pd.DataFrame(data_table[1:], columns=data_table[0])  # convert output nested list to a dataframe
print(output_df)

for col in column_list:  # create a histogram plot for each column
    fig, ax = plt.subplots()  # create a figure
    values_dict = getMostCommon(df[col])[1]  # take the element frequency dictionary from the function
    x = np.arange(len(list(set(list(values_dict.keys())))))  # define the numerical range for the x-axis
    ax.bar(x, list(values_dict.values()))  # plot the bar plot
    # set axes labels and the topic for the plot
    ax.set_xlabel('Row Elements')
    ax.set_title('Histogram bar chart for "' + col + '" column')
    ax.set_ylabel('Frequency')
    # change the numerical x-axis labels to the column values
    plt.xticks(x, tuple(set(list(values_dict.keys()))))
    plt.xticks(rotation=45)  # rotate x-axis labels
    plt.show()  # show the plot as a figure
