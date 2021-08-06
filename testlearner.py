""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import math  		  	   		   	 			  		 			 	 	 		 		 	
import sys
import matplotlib.pyplot as plt
import time
  		  	   		   	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import LinRegLearner as lrl
import DTLearner as dtl
import BagLearner as bl
import RTLearner as rt
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    if len(sys.argv) != 2:  		  	   		   	 			  		 			 	 	 		 		 	
        print("Usage: python testlearner.py <filename>")  		  	   		   	 			  		 			 	 	 		 		 	
        sys.exit(1)  		  	   		   	 			  		 			 	 	 		 		 	
    inf = open(sys.argv[1])

    if sys.argv[1] == 'Data/Istanbul.csv':
        data = np.genfromtxt(inf, delimiter=",")
        data = data[1:, 1:]
        np.random.seed(903650161)
        np.random.shuffle(data)

    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
        np.random.seed(903650161)
        np.random.shuffle(data)
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
  		  	   		   	 			  		 			 	 	 		 		 	
    # compute how much of the data is training and testing  		  	   		   	 			  		 			 	 	 		 		 	
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 			  		 			 	 	 		 		 	
    test_rows = data.shape[0] - train_rows  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # separate out training and testing data  		  	   		   	 			  		 			 	 	 		 		 	
    train_x = data[:train_rows, 0:-1]  		  	   		   	 			  		 			 	 	 		 		 	
    train_y = data[:train_rows, -1]  		  	   		   	 			  		 			 	 	 		 		 	
    test_x = data[train_rows:, 0:-1]  		  	   		   	 			  		 			 	 	 		 		 	
    test_y = data[train_rows:, -1]  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"{test_x.shape}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"{test_y.shape}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # create a learner and train it  		  	   		   	 			  		 			 	 	 		 		 	
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		   	 			  		 			 	 	 		 		 	
    learner.add_evidence(train_x, train_y)  # train it  		  	   		   	 			  		 			 	 	 		 		 	
    print(learner.author())  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # evaluate in sample  		  	   		   	 			  		 			 	 	 		 		 	
    pred_y = learner.query(train_x)  # get the predictions  		  	   		   	 			  		 			 	 	 		 		 	
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print("In sample results")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"RMSE: {rmse}")  		  	   		   	 			  		 			 	 	 		 		 	
    c = np.corrcoef(pred_y, y=train_y)  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"corr: {c[0,1]}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # evaluate out of sample  		  	   		   	 			  		 			 	 	 		 		 	
    pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 			  		 			 	 	 		 		 	
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print("Out of sample results")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"RMSE: {rmse}")  		  	   		   	 			  		 			 	 	 		 		 	
    c = np.corrcoef(pred_y, y=test_y)  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"corr: {c[0,1]}")  		  	   		   	 			  		 			 	 	 		 		 	

    #experiment 1

    rmse_outsample = []
    rmse_insample = []
    for l_size in range(1,51,2):

        learner = dtl.DTLearner(leaf_size = l_size, verbose=True)  # create a LinRegLearner
        learner.add_evidence(train_x, train_y)

        #evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_outsample.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))
        #evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_insample.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))

    leafSize = list(range(1,51,2))
    chart1 = plt.plot(leafSize,rmse_outsample)
    chart2 = plt.plot(leafSize, rmse_insample)
    plt.title('Experiment 1: RMSE vs leaf size for decision tree')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.legend(['out of sample', 'in sample'])
    plt.savefig('figure1')
    plt.close()

    # Experiment 2

    rmse_outsample = []
    rmse_insample = []
    for l_size in range(1, 51,2):
        learner = bl.BagLearner(learner= dtl.DTLearner, kwargs ={'leaf_size':l_size}, bags=20 , boost= False, verbose=False)  # create a LinRegLearner
        learner.add_evidence(train_x, train_y)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_outsample.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_insample.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))

    leafSize = list(range(1, 51,2))
    chart1 = plt.plot(leafSize, rmse_outsample)
    chart2 = plt.plot(leafSize, rmse_insample)
    plt.title('Experiment 2: RMSE vs leaf size for BagLearner with bag size = 20')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.legend(['out of sample', 'in sample'])
    plt.savefig('figure2')
    plt.close()

    #experiment 3
    # Mean absolute error
    dt_result = []
    rt_result = []
    # training time
    train_time_dt = []
    train_time_rt = []
    for l_size in range(1, 51,2):
        #decision tree
        dt_learner = dtl.DTLearner(leaf_size=l_size, verbose=True)  # create a LinRegLearner
        start = time.time()
        dt_learner.add_evidence(train_x, train_y)
        train_time_dt.append(time.time()-start)
        pred_y = dt_learner.query(test_x)  # get the predictions
        dt_result.append(np.abs(test_y - pred_y).sum() / test_y.shape[0])

        # random tree
        rt_learner = rt.RTLearner(leaf_size=l_size, verbose=True)  # create a LinRegLearner
        start = time.time()
        rt_learner.add_evidence(train_x, train_y)
        train_time_rt.append(time.time() - start)

        pred_y = rt_learner.query(test_x)  # get the predictions
        rt_result.append(np.abs(test_y - pred_y).sum() / test_y.shape[0])

    leafSize = list(range(1, 51, 2))
    chart1 = plt.plot(leafSize, dt_result)
    chart2 = plt.plot(leafSize, rt_result)
    plt.title('Experiment 3: Mean absolute error vs leaf size for decision tree and random tree', fontsize=10)
    plt.xlabel('leaf size')
    plt.ylabel('Mean Absolute Error')
    plt.legend(['decision tree', 'random tree'])
    plt.savefig('figure3')
    plt.close()

    #training time
    chart1 = plt.plot(leafSize, train_time_dt)
    chart2 = plt.plot(leafSize, train_time_rt)
    plt.title('Experiment 3: Training time vs leaf size for decision tree and random tree', fontsize=10)
    plt.xlabel('leaf size')
    plt.ylabel('training time')
    plt.legend(['decision tree', 'random tree'])
    plt.savefig('figure4')



