import numpy as np
import LinRegLearner as linReg
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners =[]
    def author(self):
        return "samoozegar3"
    def add_evidence(self, X, Y):
        bagCount = 20
        for i in range(bagCount):
            learner = bl.BagLearner(learner = linReg.LinRegLearner, kwargs ={}, bags=bagCount , boost= False, verbose=False)
            learner.add_evidence(X,Y)
            self.learners.append(learner)
    def query(self, X_test):
        predicted = [single_learner.query(X_test) for single_learner in self.learners]
        return np.mean(predicted, axis=0)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
