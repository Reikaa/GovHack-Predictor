__author__ = 'Thush'

import csv
import pickle

class DataManipulation(object):

    def __init__(self):
        self.rows = []
        self.cancers = []

    def load_data(self):

        rowCount = -1
        with open('data.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            prev_gender = ''
            for row in reader:

                rowCount = rowCount+1
                if rowCount==0:
                    continue
                if row[0]>='1968' and row[0]<='1981':
                    continue
                if row[1]=='Persons':
                    continue
                if row[3] not in self.cancers:
                    self.cancers.append(row[3])

                if not row[1] == prev_gender:
                    self.rows.append([])

                self.rows[-1].append(row)
                prev_gender = row[1]
            print 'done'

    def get_max_patients(self, rows_k):
        arr = []
        for row in rows_k:
            arr.append(float(row[23]));

        return max(arr)

    def get_input_vectors(self, rows, cancers, t, s):

        all_inputs = []
        all_outputs = []

        for k in xrange(len(rows)):
            max_patients = self.get_max_patients(rows[k])
            for i in xrange(len(rows[k])-t-s):
                input = []
                output = []
                cancer_val = cancers.index(rows[k][i][3])
                cancer_val = float(cancer_val*1.0/len(cancers))
                input.append(cancer_val)

                if rows[k][i][1]=='Male':
                    input.append(1)
                elif rows[k][i][1]=='Female':
                    input.append(0)

                for j in xrange(t):
                    if(max_patients>0.0):
                        input.append(float(rows[k][i+j][23])/max_patients)
                    else:
                        input.append(float(rows[k][i+j][23]))

                for j in xrange(s):
                    if max_patients >0.0:
                        output.append(float(rows[k][i+t+j][23])/max_patients)
                    else:
                        output.append(float(rows[k][i+t+j][23]))

                all_inputs.append(input)
                all_outputs.append(output)
        return [all_inputs,all_outputs]


if __name__ == '__main__':
    d = DataManipulation()
    d.load_data()
    [all_in,all_out] = d.get_input_vectors(d.rows,d.cancers,6,3)
    pickle.dump([all_in,all_out], open("data.pkl", "wb"))