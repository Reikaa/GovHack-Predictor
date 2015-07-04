__author__ = 'Thush'

import csv
import pickle
import numpy as np

class DataManipulation(object):

    def __init__(self):
        self.rows = []
        self.cancers = []
        self.max_pat = []
        self.pred_inputs = []
    def load_data(self):

        rowCount = -1
        with open('data.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            prev_gender = ''
            prev_status = ''
            for row in reader:

                rowCount = rowCount+1
                if rowCount==0:
                    continue
                if (row[0]>='1968' and row[0]<='1981') or row[0]=='2011':
                    continue
                if row[1]=='Persons':
                    continue
                if row[4]=='':
                    continue

                if row[3] not in self.cancers:
                    self.cancers.append(row[3])

                if not (row[1] == prev_gender or row[2] == prev_status) :
                    self.rows.append([])

                self.rows[-1].append(row)
                prev_gender = row[1]
                prev_status = row[2]
            print 'done'

    def get_max_patients(self, rows_k):
        arr = []

        for row in rows_k:
            arr.append(float(row[23]))
        return max(arr)

    def get_input_vectors(self, rows, cancers, t, s):

        all_inputs = []
        all_outputs = []
        all_valid_ins = []
        all_valid_outs = []
        all_test_ins = []
        all_test_outs = []

        for k in xrange(len(rows)):
            max_patients = self.get_max_patients(rows[k])
            for i in xrange(len(rows[k])-t-s):
                input = []
                output = []
                cancer_val = cancers.index(rows[k][i][3])
                cancer_val = float(cancer_val*1.0/len(cancers))
                input.append(cancer_val)

                if rows[k][i][1]=='Male':
                    input.append(1.0)
                elif rows[k][i][1]=='Female':
                    input.append(0.0)

                if rows[k][i][2]=='Incidence':
                    input.append(0.0)
                elif rows[k][i][2]=='Mortality':
                    input.append(1.0)

                for j in xrange(t):
                    if(max_patients>0.0):
                        input.append(float(rows[k][i+j][23])/max_patients)
                    else:
                        input.append(float(rows[k][i+j][23]))

                for j in xrange(s):
                    if max_patients>0.0:
                        output.append(float(rows[k][i+t+j][23])/max_patients)
                    else:
                        output.append(float(rows[k][i+t+j][23]))

                if len(output)<5 or len(input)<9:
                    print 'problem'

                num_valid = 5
                num_test = 10
                if i<len(rows[k])-t-s-(num_valid+num_test):
                    all_inputs.append(input)
                elif i<len(rows[k])-t-s-num_test:
                    all_valid_ins.append(input)
                else:
                    all_test_ins.append(input)

                if i<len(rows[k])-t-s-(num_valid+num_test):
                    all_outputs.append(output)
                elif i<len(rows[k])-t-s-num_test:
                    all_valid_outs.append(output)
                else:
                    all_test_outs.append(output)

            self.max_pat.append([all_inputs[-1][0],all_inputs[-1][1],all_inputs[-1][2],max_patients])
        print np.asarray(all_inputs).shape
        print np.asarray(all_outputs).shape
        print np.asarray(all_valid_ins).shape
        print np.asarray(all_valid_outs).shape
        print np.asarray(all_test_ins).shape
        print np.asarray(all_test_outs).shape

        return [all_inputs,all_outputs,all_valid_ins,all_valid_outs,all_test_ins,all_test_outs]

    def get_pred_inputs(self, rows, cancers, t):
        pred_inputs=[]
        for k in xrange(len(rows)):
            max_patients = self.get_max_patients(rows[k])
            i = len(rows[k])-t
            input = []
            cancer_val = cancers.index(rows[k][i][3])
            cancer_val = float(cancer_val*1.0/len(cancers))
            input.append(cancer_val)

            if rows[k][i][1]=='Male':
                input.append(1.0)
            elif rows[k][i][1]=='Female':
                input.append(0.0)

            if rows[k][i][2]=='Incidence':
                input.append(0.0)
            elif rows[k][i][2]=='Mortality':
                input.append(1.0)

            for j in xrange(t):
                if(max_patients>0.0):
                    input.append(float(rows[k][i+j][23])/max_patients)
                else:
                    input.append(float(rows[k][i+j][23]))

            pred_inputs.append(input)

        return pred_inputs

if __name__ == '__main__':
    d = DataManipulation()
    d.load_data()
    [all_in,all_out,all_valid_in,all_valid_out,all_test_in,all_test_out] = d.get_input_vectors(d.rows,d.cancers,6,5)
    pred_inputs = d.get_pred_inputs(d.rows,d.cancers,6)
    pickle.dump([all_in,all_out,all_valid_in,all_valid_out,all_test_in,all_test_out], open("data.pkl", "wb"))
    pickle.dump(d.max_pat, open("max_patients.pkl","wb"))
    pickle.dump(pred_inputs, open("pred_ins.pkl","wb"))
    pickle.dump(d.cancers, open("cancers.pkl","wb"))