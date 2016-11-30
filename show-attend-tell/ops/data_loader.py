import csv
import os

training_data = dict()
testing_data = dict()

train_labels = 'data_dir/train_labels.csv'
testing_labels = 'data_dir/test_labels.csv'

if os.path.isfile(train_labels):
    with open(train_labels) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            training_data[row[0]]=row[1]

if os.path.isfile(testing_labels):
    with open(testing_labels) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            testing_data[row[0]]=row[1]
