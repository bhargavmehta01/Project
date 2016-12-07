import sys
sys.path.append("..")
import operator
import time
from math import sqrt
from svmutil import *

block_counter = 0
label_sum = 0
label_square = 0
labels_arr = []
label_calc = 0
column_correlation_value = {}
selected_cols = []
num_features = 0


# Module To Read Labels ##
def read_labels(label_file):
    f = open(label_file)
    l = f.readline()
    while l != '':
        a = l.split()
        labels_arr.append(int(a[0]))
        l = f.readline()
    f.close()


# Module To Read Data #
def read_data(data_file):
    global block_counter, selected_cols
    f = open(data_file)
    data = []
    l = f.readline()

    while l != '':
        a = l.split()
        l2 = []
        if len(selected_cols) > 0:
            for col_number in selected_cols:
                l2.append(int(a[col_number]))
        else:
            for j in range(block_counter, len(a), 1):
                l2.append(int(a[j]))
        data.append(l2)
        l = f.readline()
    f.close()
    return data


# Module To Calculate Pearson Correlation #
def pearson_correlation(feature):
    feature_sum = sum(feature)
    feature_square = sum([n * n for n in feature])
    size = len(feature)
    product_sum = 0
    for i in range(0, size, 1):
        product_sum += (labels_arr[i] * feature[i])
    numerator = product_sum - ((label_sum * feature_sum) / size)
    feature_calc = (feature_square - (feature_sum * feature_sum) / size)
    denominator = sqrt(label_calc * feature_calc)
    if denominator == 0:
        return 0
    return abs(numerator / denominator)


# Module To Get Sum And Sum Of Squares Of Labels #
def labels_info():
    global label_sum, label_square, label_calc
    size = len(labels_arr)
    label_sum = sum(labels_arr)
    label_square = sum([n * n for n in labels_arr])
    label_calc = (label_square - (label_sum * label_sum) / size)


# Start of Main Method #
def main(args):
    global selected_cols, num_features
    start_time = time.time();
    print("** Reading Train Data Labels **")
    label_file = sys.argv[1]
    read_labels(label_file)
    print("** Reading Of Train Data Labels Completed **")
    labels_info()
    print("** Reading of Train Data Started **")
    data_file = sys.argv[2]
    data = read_data(data_file)
    print("** Reading of Train Data Completed **")
    rows = len(data)
    columns = len(data[0])
    column_counter = 0

    print("** Performing Feature Reduction using Pearson Correlation **")

    while column_counter < columns:
        feature = []
        for i in range(0, rows, 1):
            feature.append(data[i][column_counter])
        column_correlation_value[column_counter] = pearson_correlation(feature)
        column_counter += 1

    sorted_correlation = sorted(column_correlation_value.items(), key=operator.itemgetter(1))
    len_sorted_correlation = len(sorted_correlation) - 1

    cols = []
    num_features += 10
    print("Number of Features Selected: " + str(num_features))
    for i in range(len_sorted_correlation, len_sorted_correlation - num_features, -1):
        cols.append(sorted_correlation[i][0])
    selected_cols = sorted(cols, key=int)
    print(selected_cols)
    reduced_data = read_data(data_file)

    print("** Feature Reduction Completed **")

    print("** Building Model **")
    model = svm_train(labels_arr, reduced_data)
    train_labels, train_acc, train_vals = svm_predict(labels_arr, reduced_data, model)
    acc, mse, scc = evaluations(labels_arr, train_labels)
    print("** Model Built Successfully **")

    print("** Validation results for train data **")
    print("accuracy :: " + str(acc))
    print("mse :: " + str(mse))
    print("scc :: " + str(scc))
    print("*********************")

    print("** Reading of Test Data Started **")
    test_file = sys.argv[3]
    test_data = read_data(test_file)

    print("** Reading of Test Data Completed **")

    predict_labels, predict_acc, predict_val = svm_predict([0] * len(test_data), test_data, model)

    print("** Predicted Output Labels Are On /Output/Prediction.txt **")

    elapsed_time = time.time() - start_time
    print("Total Execution Time:: " + str(elapsed_time) + " seconds.")

    sys.stdout = open("Output/Prediction.txt", "w")
    for i in range(0, len(predict_labels), 1):
        #print(int(predict_labels[i]))
        print(str(int(predict_labels[i])))

    print("***Prediction Complete For Test Data*")




if __name__ == "__main__": main(sys.argv)
