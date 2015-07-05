__author__ = 'Thushan Ganegedara'

import numpy as np
from SparseAutoencoderGPU import SparseAutoencoder
from OutputLayerGPU import OutputLayer

from scipy import optimize
from scipy import misc
from numpy import linalg as LA

import os
from PIL import Image
from numpy import linalg as LA
from math import sqrt
import gzip,cPickle

from theano import function, config, shared, sandbox, Param
import theano.tensor as T
import time

import sys,getopt

from utils import tile_raster_images
import csv
try:
    import PIL.Image as Image
except ImportError:
    import Image

class StackedAutoencoder(object):


    def __init__(self,in_size=8, hidden_size = [500, 500, 250], out_size = 10, batch_size = 10, corruption_levels=[0.1, 0.1, 0.1],dropout=True,drop_rates=[0.5,0.2,0.2]):
        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations_train = []
        self.sa_activations_test = []
        self.thetas = []
        self.thetas_as_blocks = []

        self.dropout = dropout
        self.drop_rates = drop_rates

        #check if there are layer_count+1 number of dropout rates (extra one for softmax)
        if dropout:
            assert self.n_layers+1 == len(self.drop_rates)

        self.corruption_levels = corruption_levels

        #check if there are layer_count number of corruption levels
        if denoising:
            assert self.n_layers == len(self.corruption_levels)

        self.cost_fn_names = ['sqr_err', 'neg_log']

        self.x = T.matrix('x')  #store the inputs
        self.y = T.matrix('y') #store the labels for the corresponding inputs

        self.fine_cost = T.dscalar('fine_cost') #fine tuning cost
        self.error = T.dscalar('test_error')    #test error value

        #print network info
        print "Network Info:"
        print "Layers: %i" %self.n_layers
        print "Layer sizes: ",
        print self.h_sizes
        print ""
        print "Building the model..."

        #intializing the network.
        #crating SparseAutoencoders and storing them in sa_layers
        #calculating hidden activations (symbolic) and storing them in sa_activations_train/test
        #there are two types of activations as the calculations are different for train and test with dropout
        for i in xrange(self.n_layers):

            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            #if i==0 input is the raw input
            if i==0:
                curr_input_train = self.x
                curr_input_test = self.x
            #otherwise input is the previous layer's hidden activation
            else:
                a2_train = self.sa_layers[-1].get_hidden_act(training=True)
                a2_test = self.sa_layers[-1].get_hidden_act(training=False)
                self.sa_activations_train.append(a2_train)
                self.sa_activations_test.append(a2_test)
                curr_input_train = self.sa_activations_train[-1]
                curr_input_test = self.sa_activations_test[-1]

            sa = SparseAutoencoder(n_inputs=curr_input_size, n_hidden=self.h_sizes[i],
                                   x_train=curr_input_train, x_test=curr_input_test,
                                   dropout=dropout, dropout_rate=self.drop_rates[i])
            self.sa_layers.append(sa)
            self.thetas.extend(self.sa_layers[-1].get_params())
            self.thetas_as_blocks.append(self.sa_layers[-1].get_params())

        #-1 index gives the last element
        a2_train = self.sa_layers[-1].get_hidden_act(training=True)
        a2_test = self.sa_layers[-1].get_hidden_act(training=False)
        self.sa_activations_train.append(a2_train)
        self.sa_activations_test.append(a2_test)

        self.outLayer = OutputLayer(n_inputs=self.h_sizes[-1], n_outputs=self.o_size,
                                         x_train=self.sa_activations_train[-1], x_test = self.sa_activations_test[-1],
                                         y=self.y, dropout=self.dropout, dropout_rate=self.drop_rates[-1])
        self.lam_fine_tune = T.scalar('lam')
        self.fine_cost = self.outLayer.get_cost(self.lam_fine_tune,cost_fn=self.cost_fn_names[1])

        self.thetas.extend(self.outLayer.theta)

        #measure test performance
        self.error = self.outLayer.get_error(self.y)
        self.predict = self.outLayer.get_output()


    def load_max_pat(self,file_path):
        f = open(file_path, 'rb')
        max_patients = cPickle.load(f)

        return max_patients

    def load_pred_ins(self,file_path):
        f = open(file_path, 'rb')
        pred_ins = cPickle.load(f)

        return shared(value=np.asarray(pred_ins,dtype=config.floatX),borrow=True)

    def load_cancer_data(self,file_path):
        f = open(file_path,'rb')
        cancer_data = cPickle.load(f)
        return cancer_data

    def load_data(self,file_path='data.pkl',make_predict=True):

        f = open(file_path, 'rb')
        if not make_predict:
            all_ins,all_outs,all_v_in,all_v_out,all_t_in,all_t_out = cPickle.load(f)
            train_set = [all_ins,all_outs]
            valid_set = [all_v_in,all_v_out]
            test_set = [all_t_in,all_t_out]
            f.close()
        else:
            all_ins,all_outs,all_v_in,all_v_out = cPickle.load(f)
            train_set = [all_ins,all_outs]
            valid_set = [all_v_in,all_v_out]
            test_set = [all_v_in,all_v_out]

        def get_shared_data(data_xy):
            data_x,data_y = data_xy
            shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
            shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

            return shared_x,shared_y


        train_x,train_y = get_shared_data(train_set)
        valid_x,valid_y = get_shared_data(valid_set)
        test_x,test_y = get_shared_data(test_set)


        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x,test_y)]

        return all_data

    def greedy_pre_training(self, train_x, batch_size=1, pre_lr=0.25,denoising=False):

        pre_train_fns = []
        index = T.lscalar('index')
        lam = T.scalar('lam')
        beta = T.scalar('beta')
        rho = T.scalar('rho')

        i = 0
        print "\nCompiling functions for DA layers..."
        for sa in self.sa_layers:


            cost, updates = sa.get_cost_and_updates(l_rate=pre_lr, lam=lam, beta=beta, rho=rho, cost_fn=self.cost_fn_names[1],
                                                    corruption_level=self.corruption_levels[i], denoising=denoising)

            #the givens section in this line set the self.x that we assign as input to the initial
            # curr_input value be a small batch rather than the full batch.
            # however, we don't need to set subsequent inputs to be an only a minibatch
            # because if self.x is only a portion, you're going to get the hidden activations
            # corresponding to that small batch of inputs.
            # Therefore, setting self.x to be a mini-batch is enough to make all the subsequents use
            # hidden activations corresponding to that mini batch of self.x
            sa_fn = function(inputs=[index, Param(lam, default=0.25), Param(beta, default=0.25), Param(rho, default=0.2)], outputs=cost, updates=updates, givens={
                self.x: train_x[index * batch_size: (index+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)
            i = i+1

        return pre_train_fns

    def fine_tuning(self, datasets, batch_size=1, fine_lr=0.2):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.fine_cost, self.thetas)

        updates = [(param, param - gparam*fine_lr)
                   for param, gparam in zip(self.thetas,gparams)]

        fine_tuen_fn = function(inputs=[index, Param(self.lam_fine_tune,default=0.25)],outputs=self.fine_cost, updates=updates, givens={
            self.x: train_set_x[index * self.batch_size: (index+1) * self.batch_size],
            self.y: train_set_y[index * self.batch_size: (index+1) * self.batch_size]
        })

        validation_fn = function(inputs=[index],outputs=self.error, givens={
            self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },name='valid')

        def valid_score():
            return [validation_fn(i) for i in xrange(n_valid_batches)]
        return fine_tuen_fn, valid_score

    def train_model(self, datasets=None, pre_epochs=5, fine_epochs=300, pre_lr=0.25, fine_lr=0.2, batch_size=1, lam=0.0001, beta=0.25, rho = 0.2,denoising=False):

        print "Training Info..."
        print "Batch size: ",
        print batch_size
        print "Pre-training: %f (lr) %i (epochs)" %(pre_lr,pre_epochs)
        print "Fine-tuning: %f (lr) %i (epochs)" %(fine_lr,fine_epochs)
        print "Corruption: ",
        print denoising,
        print self.corruption_levels
        print "Weight decay: ",
        print lam
        print "Dropout: ",
        print self.dropout,
        print self.drop_rates
        print "Sparcity: ",
        print "%f (beta) %f (rho)" %(beta,rho)

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x, batch_size=self.batch_size,pre_lr=pre_lr,denoising=denoising)

        start_time = time.clock()
        for i in xrange(self.n_layers):

            print "\nPretraining layer %i" %i
            for epoch in xrange(pre_epochs):
                c=[]
                for batch_index in xrange(n_train_batches):
                    c.append(pre_train_fns[i](index=batch_index, lam=lam, beta=beta, rho=rho))

                print 'Training epoch %d, cost ' % epoch,
                print np.mean(c)

            end_time = time.clock()
            training_time = (end_time - start_time)

            print "Training time: %f" %training_time

        #########################################################################
        #####                          Fine Tuning                          #####
        #########################################################################
        print "\nFine tuning..."

        fine_tune_fn,valid_model = self.fine_tuning(datasets,batch_size=self.batch_size,fine_lr=fine_lr)


        #########################################################################
        #####                         Early-Stopping                        #####
        #########################################################################
        patience = 10 * n_train_batches # look at this many examples
        patience_increase = 2.
        improvement_threshold = 1.005
        #validation frequency - the number of minibatches to go through before checking validation set
        validation_freq = min(n_train_batches,patience/2)

        #we want to minimize best_valid_loss, so we shoudl start with largest
        best_valid_loss = np.inf
        test_score = 0.

        done_looping = False
        epoch = 0

        while epoch < fine_epochs and (not done_looping):
            epoch = epoch + 1
            fine_tune_cost = []
            for mini_index in xrange(n_train_batches):
                cost = fine_tune_fn(index=mini_index,lam=lam)
                fine_tune_cost.append(cost)
                #what's the role of iter? iter acts as follows
                #in first epoch, iter for minibatch 'x' is x
                #in second epoch, iter for minibatch 'x' is n_train_batches + x
                #iter is the number of minibatches processed so far...
                iter = (epoch-1) * n_train_batches + mini_index

                # this is an operation done in cycles. 1 cycle is iter+1/validation_freq
                # doing this every epoch
                if (iter+1) % validation_freq == 0:
                    validation_losses = valid_model()
                    curr_valid_loss = np.mean(validation_losses)
                    print 'epoch %i, minibatch %i/%i, validation error is %f %%' %(epoch, mini_index+1,n_train_batches,curr_valid_loss*100)

                    if curr_valid_loss < best_valid_loss:

                        if (
                            curr_valid_loss < best_valid_loss * improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_valid_loss = curr_valid_loss
                        best_iter = iter

            print 'Fine tune cost for epoch %i, is %f' %(epoch+1,np.mean(fine_tune_cost))
            #patience is here to check the maximum number of iterations it should check
            #before terminating
            if patience <= iter:
                done_looping = True
                break

    def get_correct_max_pat(self,x,max_pat):
        for p in max_pat:
            if p[0]==x[0] and p[1]==x[1] and p[2]==x[2]:
                return p[3]

        return -1

    def test_model(self,test_set_x,test_set_y,batch_size= 1,max_pat=None,cancers=None):

        print '\nTesting the model...'
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')

        #no update parameters, so this just returns the values it calculate
        #without objetvie function minimization
        test_fn = function(inputs=[index], outputs=[self.error,self.y,self.predict,self.x], givens={
            self.x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            self.y: test_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }, name='test')

        e=[]
        pred_vals = []
        act_vals = []

        errsAll = dict()
        for batch_index in xrange(n_test_batches):
            err,act,pred,x = test_fn(batch_index)
            e.append(err)

            for p,a,x_i in zip(pred,act,x):

                errsSingle = []
                for p2,a2 in zip(p,a):
                    max_pat_val = self.get_correct_max_pat(x_i,max_pat)
                    diff = abs(p2-a2)
                    if a2>0.0:
                        errTmp = np.mean(diff/a2)
                    else:
                        errTmp = 0.0
                    errsSingle.append(errTmp)

                key = self.get_key(x_i,cancers)

                if key not in errsAll:
                    errsAll[key]=[errsSingle]
                else:
                    errPrev = errsAll[key]
                    errPrev.append(errsSingle)
                    errsAll[key] = errPrev


                #if all(v == 0 for v in errsSingle):

            #pred_vals.append(pred)
            idx = 2
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for p in pred[idx]:
                print int(p*max_pat_val),
            print ""
            idx = 5
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for p in pred[idx]:
                print int(p*max_pat_val),
            print ""
            idx = 8
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for p in pred[idx]:
                print int(p*max_pat_val),
            print ""

            idx = 2
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for a in act[idx]:
                print int(a*max_pat_val),
            print ""
            idx = 5
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for a in act[idx]:
                print int(a*max_pat_val),
            print ""
            idx = 8
            max_pat_val = self.get_correct_max_pat(x[idx],max_pat)
            for a in act[idx]:
                print int(a*max_pat_val),
            print ""

            #act_vals.append(act)
            #print pred,',',act

        keys = []
        errors = []
        for k in errsAll:
            keys.append(k)
            tmp3 = errsAll.get(k)
            tmp = np.asarray(errsAll.get(k))
            errsForKey = np.mean(tmp*100,axis=0);
            errors.append(errsForKey)
            print 'Test Error for ', k, ": ", errsForKey

        self.create_csv_errors(keys,errors)

    def get_key(self,x,cancers):
        c_idx = int(round(x[0]*len(cancers)))
        s = cancers[c_idx]
        if x[1]==1.0:
            s = s + ",Male"
        else:
            s = s + ",Female"

        if x[2]==1.0:
            s = s + ",Mortality"
        else:
            s = s + ",Incidence"

        return s

    def predict_val(self,pred_ins):

        print 'Predicting ....'
        #no update parameters, so this just returns the values it calculate
        #without objetvie function minimization
        pred_fn = function(inputs=[], outputs=[self.predict], givens={
            self.x: pred_ins
        }, name='predict')

        e=[]
        pred_vals = []
        act_vals = []
        pred = pred_fn()

        return pred

    def create_csv_errors(self,keys,errors):
        all_strings = []
        header = ['Cancer','Gender','Status','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
        all_strings.append(header)

        for (k,e) in zip(keys,errors):
            single_str = []
            single_str.extend(k.split(","))
            for val in e:
                single_str.append(str(val))

            all_strings.append(single_str)

        with open('errors.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(all_strings)

    def create_csv(self,x,pred,cancers,max_patients,num_in_years):
        x_arr = x.get_value()
        all_strings = []

        header = []
        header.append('Cancer')
        header.append('Gender')
        header.append('Status')
        for yr in xrange(2006,2011):
            header.append(str(yr))
        for yr in xrange(2011,2021):
            header.append(str(yr))
        all_strings.append(header)

        for i in xrange(len(x_arr)):
            single_str = []
            inp = x_arr[i]
            c_idx = int(round(inp[0]*len(cancers)))
            single_str.append(cancers[c_idx])
            if inp[1]==0.0:
                single_str.append('Female')
            elif inp[1]==1.0:
                single_str.append('Male')

            if inp[2]==0.0:
                single_str.append('Incidence')
            elif inp[2]==1.0:
                single_str.append('Mortality')

            max_pat_val = self.get_correct_max_pat(inp,max_patients)

            for k in xrange(num_in_years):
                single_str.append(str(int(round(inp[k+3]*max_pat_val))))

            for j in xrange(len(pred[0][i])):
                p = pred[0][i][j]
                tmp = int(round(p*max_pat_val))
                single_str.append(str(tmp))


            all_strings.append(single_str)

        with open('results.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(all_strings)

    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)


if __name__ == '__main__':
    #sys.argv[1:] is used to drop the first argument of argument list
    #because first argument is always the filename
    try:
        opts,args = getopt.getopt(sys.argv[1:],"h:p:f:b:d:",["w_decay=","early_stopping=","dropout=","corruption=","beta=","rho="])
    except getopt.GetoptError:
        print '<filename>.py -h [<hidden values>] -p <pre-epochs> -f <fine-tuning-epochs> -b <batch_size> -d <data_folder>'
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        lam = 0.0
        dropout = True
        drop_rates = [0.1,0.1,0.1]
        corr_level = [0.1, 0.2, 0.3]
        denoising = False
        beta = 0.0
        rho = 0.0

        for opt,arg in opts:
            if opt == '-h':
                hid_str = arg
                hid = [int(s.strip()) for s in hid_str.split(',')]
            elif opt == '-p':
                pre_ep = int(arg)
            elif opt == '-f':
                fine_ep = int(arg)
            elif opt == '-b':
                b_size = int(arg)
            elif opt == '-d':
                data_dir = arg
            elif opt == '--w_decay':
                lam = float(arg)
            elif opt == '--dropout':
                drop_rate_str = arg.split(',')[0]
                if drop_rate_str=='y':
                    dropout = True
                    drop_rates = [float(s.strip()) for s in arg.split(',')[1:]]
                elif drop_rate_str == 'n':
                    dropout = False
            elif opt == '--corruption':
                corr_str = arg
                denoise_str = corr_str.split(',')[0]
                if denoise_str=='y':
                    denoising = True
                    corr_level = [float(s.strip()) for s in corr_str.split(',')[1:]]
                else:
                    denoising = False
            elif opt == '--beta':
                beta = float(arg)
            elif opt == '--rho':
                rho = float(arg)

    #when I run in Pycharm
    else:
        lam = 0.0
        hid = [225,225,225,225]
        pre_ep = 25
        fine_ep = 500
        b_size = 10
        dropout = False
        drop_rates = [0.2,0.2,0.2,0.2, 0.2]
        corr_level = [0.01, 0.01, 0.01, 0.01]
        denoising = True
        beta = 0.2
        rho = 0.2
        make_predict = True
        if make_predict:
            data_dir = 'data_pred.pkl'
        else:
            data_dir = 'data.pkl'

    sae = StackedAutoencoder(hidden_size=hid, batch_size=b_size, corruption_levels=corr_level,dropout=dropout,drop_rates=drop_rates)
    all_data = sae.load_data(data_dir,make_predict=make_predict)
    max_patients = sae.load_max_pat('max_patients.pkl')
    pred_ins = sae.load_pred_ins('pred_ins.pkl')
    cancer_data = sae.load_cancer_data('cancers.pkl')
    sae.train_model(datasets=all_data, pre_epochs=pre_ep, fine_epochs=fine_ep, batch_size=sae.batch_size, lam=lam, beta=beta, rho=rho, denoising=denoising)

    if not make_predict:
        sae.test_model(all_data[2][0],all_data[2][1],batch_size=sae.batch_size,max_pat=max_patients,cancers=cancer_data)
    else:
        pred_vals = sae.predict_val(pred_ins)
        sae.create_csv(pred_ins,pred_vals,cancer_data,max_patients,5)

    #max_inp = sae.get_input_threshold(all_data[0][0])
    #sae.visualize_hidden(max_inp)
