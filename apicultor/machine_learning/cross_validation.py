#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import time
from ..gradients.subproblem import *
from .fairness import *
from .dependency import *
from .explain import *
from random import sample
import logging
import warnings
import signal
from pathos.pools import ParallelPool as Pool

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def acc_score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return accuracy_score(targs, classes, sample_weight=lw)


def score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return mean_squared_error(targs, classes, sample_weight=lw)

# it's going to work with MEM DSVM implementation
class DiscriminationError(Exception):
    def __init__(self):
        super().__init__("Instance violates demographic parity")
        
class DiscriminationError(Exception):
    def __init__(self,e):
        super().__init__(("Something happened!",e,"Continuing"))

def hard_kill_pool(pids, pt):
    #terminates a function
    for pid in pids:
        os.kill(pid, signal.SIGINT)
    #terminates a tree of processes
    pt.terminate()        
        
def productionize(model,features_train,targets_train,features_test,targets_test,features,targets,C,reg,k0,k1,last,criteria,intersects,logical,track_conflict):
    try:
        training_time = time.time()
        model.fit_model(features_train, targets_train, k0, k1,
                                C, reg, 1./features_train.shape[1], 0.8)
        training_time = np.abs(training_time - time.time())
        #timing['training_times'][i].append(training_time)
        #print("Training time is: ", training_time)
        clf_predictions_train = model.predictions(features_train, targets_train)
        train_score = score(targets_train, clf_predictions_train)
        # Train statistical parity
        protection_val_rule_train = p_rule(clf_predictions_train, targets_train, model.w, features_train, model.proba)
        if type(protection_val_rule_train) != bool and protection_val_rule_train >= .8:
            #print('Statistical parity at instance training is',protection_val_rule_train)
            parity_train = protection_val_rule_train
        else:
            return DiscriminationError
        pex, cex, vis = explain(model, features_train, targets_train, criteria, intersects, logical)
        # depends on linked classes
        #print("Train parent explanation: ", pex)
        # a subgroup
        #print("Train child explanation: ", cex)
        #print("Train BTC: ", BTC(targets_train, clf_predictions_train))
        bec, cons = BEC(targets_train, clf_predictions_train)
        #print("Train BEC: ", bec)
        #print("Train Scoring Error is: ",train_score)
        clf_predictions_test = model.predictions(
                    features_test, targets_test)
        test_score = score(targets_test, clf_predictions_test)
        # Test statistical parity
        protection_val_rule_test = p_rule(
                    clf_predictions_test, targets_test, model.w, features_test, model.proba)
        if type(protection_val_rule_test) != bool and protection_val_rule_test >= .8:
            #print('Statistical parity at instance testing is',
            #              protection_val_rule_test)
            parity_test = protection_val_rule_test
        else:
            return DiscriminationError
        pex, cex, vis = explain(model, features_test, targets_test, criteria, intersects, logical)
        #print("Test parent explanation: ", pex)
        #print("Test child explanation: ", cex)
        #print("Test BTC: ", BTC(targets_test, clf_predictions_test))
        bec, cons = BEC(targets_test, clf_predictions_test)
        #print("Test BEC: ", bec)
        #print("Test Scoring Error is: ",test_score)
        clf_time = time.time()
        # apply weights after having found a fit for previous config
        if last is True:
            model.apply(model.best_layer[-1])
        model.fit_model(features, targets, kernel_configs[i][0], kernel_configs[i][1],
                                C, reg, 1./features.shape[1], 0.8)
        model.written = False
        # undo apply for later
        clf_time = np.abs(clf_time - time.time())
        #print("Classification time is: ", clf_time)
        clf_predictions = model.predictions(features, targets)
        # Grid statistical parity
        protection_val_rule = p_rule(clf_predictions, targets, model.w, features, model.proba)
        if type(protection_val_rule) != bool and protection_val_rule >= .8:
            #print('Statistical parity at instance is',protection_val_rule)
            parity = protection_val_rule
        else:
            #print("Low parity in instance")
            return DiscriminationError
        mse = score(targets, clf_predictions)
        pex, cex, vis = explain(model, features, targets, criteria, intersects, logical)
        #print("Parent search explanation: ", pex)
        #print("Child search explanation: ", cex)
        if track_conflict == None:
            #print("BTC: ", BTC(targets, clf_predictions))
            bec, cons = BEC(targets, clf_predictions)
            #print("BEC: ", bec)
        else:
            btc = BTC(targets, clf_predictions)
            bec, cons = BEC(targets, clf_predictions, track_conflict, True)
            #print("BTC: ", btc)
            #print("BEC: ", bec)
        #print("Scoring error is: ",mse)
        return mse, train_mse, test_mse, model, cons, training_time, clf_time, train_parity, test_parity, parity
    except Exception as e:
        return ProductionizationError(logger.exception(e))

def GridSearch(model, features, targets, Cs, reg_params, kernel_configs, last, criteria, intersects, logical, track_conflict=None,p_thresh=1e-4):
    """
    Perform Cross-Validation using search to find out which is the best configuration for a layer. Ideally a full cross-validation process would need at least +4000 configurations to find a most suitable, but here we are setting parameters rationally to set accurate values. The MEM uses an automatic value for gamma of 1./n_features.  
    :param model: predictive targets
    :param features: predicted targets
    :param targets: predictive targets
    :param Cs: list of C configurations
    :param reg_params: list of reg_param configurations
    :param kernel_configs: kernels configurations
    :param intersects (type(intersects) == list): list of feature column(s) to explain
    :param logical (type(logical) == list): list of bool values expressing logical observations
    :returns:                                                                                                         
      - the best estimator values (its accuracy score, its C and its reg_param value)
    """
    n = len(features)    
    features_train = []
    features_test = []
    targets_train = []
    targets_test = []
    if not targets[0].size > 0:
        for i in range(len(np.unique(targets))):
            n = len(features[np.where(targets == i)])
            test_n = int((20 * n)/100)
            train_n = n - test_n
            features_train.append(features[targets == i][:train_n])
            features_test.append(features[targets == i][train_n:])
            targets_train.append(targets[targets == i][:train_n])
            targets_test.append(targets[targets == i][train_n:])
        features_test = np.vstack(features_test)
        features_train = np.vstack(features_train)
        targets_train = np.hstack(targets_train)
        targets_test = np.hstack(targets_test)
        print(np.unique)
    else:
        n = len(features)
        test_n = int((20 * n)/100)
        train_n = n - test_n
        features_train.append(features[:train_n])
        features_train = features_train[0]
        features_test.append(features[train_n:])
        features_test = features_test[0]
        targets_train.append(targets[:train_n])
        targets_train = targets_train[0]
        targets_test.append(targets[train_n:])
        targets_test = targets_test[0]
    params = defaultdict(list)
    timing = defaultdict(list)
    scores = defaultdict(list)
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    best_estimator = defaultdict(list)
    models = defaultdict(list)
    grid_conflicts = []
    [best_estimator['score'].append([]) for i in range(len(kernel_configs))]
    [best_estimator['C'].append([]) for i in range(len(kernel_configs))]
    [best_estimator['reg_param'].append([])
     for i in range(len(kernel_configs))]
    [scores['scorings'].append([]) for i in range(len(kernel_configs))]
    [models['models'].append([]) for i in range(len(kernel_configs))]
    [train_scores['scorings'].append([]) for i in range(len(kernel_configs))]
    [test_scores['scorings'].append([]) for i in range(len(kernel_configs))]
    [timing['training_times'].append([]) for i in range(len(kernel_configs))]
    [timing['classifier times'].append([]) for i in range(len(kernel_configs))]
    [train_scores['demographic_parity'].append(
        []) for i in range(len(kernel_configs))]
    [test_scores['demographic_parity'].append(
        []) for i in range(len(kernel_configs))]
    [scores['demographic_parity'].append([])
     for i in range(len(kernel_configs))]
    params['C'].append(Cs)
    params['reg_param'].append(reg_params)
    for i in range(len(kernel_configs)):
        for j in range(len(params['C'][0])):
            try:
                training_time = time.time()              
                if last is True:
                    if len(model.best_layer) > 0:
                        model.apply(model.best_layer[-1])  
                model.gamma = .8        
                model.fit_model(features_train[:len(targets_train)], targets_train, kernel_configs[i][0], kernel_configs[
                                i][1], params['C'][0][j], params['reg_param'][0][j], 1./features_train.shape[1], 0.8)
                training_time = np.abs(training_time - time.time())
                timing['training_times'][i].append(training_time)
                #print(str().join(("Training time is: ", str(training_time))))
                if not targets_train[0].size > 1:
                    clf_predictions_train = model.predictions(
                        features_train, targets_train)
                    train_scores['scorings'][i].append(
                        score(targets_train, clf_predictions_train))
                    # Train statistical parity    
                    protection_val_rule_train = p_rule(
                        clf_predictions_train, targets_train, model.w, features_train, model.proba,p_thresh)
                else:
                    clf_predictions_train = model.w[0] * features_train + model.bias
                    train_scores['scorings'][i].append(
                        mean_squared_error(targets_train, clf_predictions_train))
                    model.proba = sigmoid(model.w[0] * features_train + model.bias)    
                    # Train statistical parity    
                    protection_val_rule_train = p_rule(
                        clf_predictions_train.T, targets_train.T, model.w[0], features_train, model.proba,p_thresh)                    
                if type(protection_val_rule_train) != bool and protection_val_rule_train >= .8:
                    print('Statistical parity at instance training is',
                          protection_val_rule_train)
                    train_scores['demographic_parity'][i].append(
                        protection_val_rule_train)
                    pass
                else:
                    print(ValueError(
                        "Discrimination and false information at training instance with parity value of", protection_val_rule_train))
                    scores['scorings'][i].append(1)
                    scores['demographic_parity'][i].append(-1)
                    models['models'][i].append(None)
                    grid_conflicts.append(
                        np.array([i for i in range(len(features))]))
                    continue
                #pex, cex, vis = explain(
                #    model, features_train, targets_train, criteria, intersects, logical)
                # depends on linked classes
                #print("Train parent explanation: ", pex)
                # a subgroup
                #print("Train child explanation: ", cex)
                #print("Training BTC with ", targets_train.shape, clf_predictions_train.shape)
                #print("Train BTC: ", BTC(targets_train, clf_predictions_train))
                bec, cons = BEC(targets_train, clf_predictions_train)
                #print("Train BEC: ", bec)
                #print(str().join(("Train Scoring Error is: ",
                #      str(train_scores['scorings'][i][j]))))
                if not targets_train[0].size > 1:
                    clf_predictions_test = model.predictions(
                        features_test, targets_test)
                    test_scores['scorings'][i].append(
                        score(targets_test, clf_predictions_test))
                    # Test statistical parity    
                    protection_val_rule_test = p_rule(
                        clf_predictions_test, targets_test, model.w, features_test, model.proba,p_thresh)
                else:
                    clf_predictions_test = model.w[0] * features_test + model.bias
                    test_scores['scorings'][i].append(
                        mean_squared_error(targets_test, clf_predictions_test))
                    model.proba = sigmoid(model.w[0] * features_test + model.bias)    
                    # Train statistical parity    
                    protection_val_rule_test = p_rule(
                        clf_predictions_test.T, targets_test.T, model.w[0], features_test, model.proba,p_thresh)                                   
                if type(protection_val_rule_test) != bool and protection_val_rule_test >= .8:
                    print('Statistical parity at instance testing is',
                          protection_val_rule_test)
                    test_scores['demographic_parity'][i].append(
                        protection_val_rule_test)
                    pass
                else:
                    print(ValueError(
                        "Discrimination and false information at test instance with parity value of", protection_val_rule_test))
                    scores['scorings'][i].append(1)
                    scores['demographic_parity'][i].append(-1)
                    models['models'][i].append(None)
                    grid_conflicts.append(
                        np.array([i for i in range(len(features))]))
                    continue
                #pex, cex, vis = explain(
                #    model, features_test, targets_test, criteria, intersects, logical)
                #print("Test parent explanation: ", pex)
                #print("Test child explanation: ", cex)
                print("Test BTC: ", BTC(targets_test, clf_predictions_test))
                bec, cons = BEC(targets_test, clf_predictions_test)
                #print("Test BEC: ", bec)
                try:
                    print(str().join(("Test Scoring Error is: ",
                      str(test_scores['scorings'][i][j]))))
                except Exception as e:
                    print(str().join(("Test Scoring Error is: ",
                      str(test_scores['scorings'][i][-1]))))
                clf_time = time.time()
                # apply weights after having found a fit for previous config
                try:
                    model.apply(model.best_layer[-1])
                except Exception as e:
                    pass
                model.fit_model(features, targets, kernel_configs[i][0], kernel_configs[i][1],
                                params['C'][0][j], params['reg_param'][0][j], 1./features.shape[1], 0.8)
                model.written = True
                # undo apply for later
                clf_time = np.abs(clf_time - time.time())
                timing['classifier times'][i].append(clf_time)
                print(str().join(("Classification time is: ", str(clf_time))))
                if not targets_train[0].size > 1:
                    clf_predictions = model.predictions(features, targets)
                    scores['scorings'][i].append(
                        score(targets, clf_predictions))
                    #Grid statistical parity    
                    protection_val_rule = p_rule(
                        clf_predictions, targets, model.w, features, model.proba,p_thresh)
                else:
                    clf_predictions = model.w[0] * features + model.bias
                    scores['scorings'][i].append(
                        mean_squared_error(targets, clf_predictions))
                    model.proba = sigmoid(model.w[0] * features + model.bias)    
                    #Grid statistical parity    
                    protection_val_rule = p_rule(
                        clf_predictions.T, targets.T, model.w[0], features, model.proba,p_thresh)                                    
                if type(protection_val_rule) != bool and protection_val_rule >= .8:
                    print('Statistical parity at instance is',
                          protection_val_rule)
                    scores['demographic_parity'][i].append(protection_val_rule)
                    pass
                else:
                    print(ValueError(
                        "Discrimination and false information at test instance with parity value of", protection_val_rule))
                    print("Broken parity is", protection_val_rule)
                    scores['scorings'][i].append(1)
                    scores['demographic_parity'][i].append(-1)
                    models['models'][i].append(None)
                    grid_conflicts.append(
                        np.array([i for i in range(len(features))]))
                    continue
                models['models'][i].append(model)
                #pex, cex, vis = explain(
                #    model, features, targets, criteria, intersects, logical)
                #print("Parent search explanation: ", pex)
                #print("Child search explanation: ", cex)
                if track_conflict == None:
                    #print("BTC: ", BTC(targets, clf_predictions))
                    bec, cons = BEC(targets, clf_predictions)
                    #print("Instance BEC: ", bec)
                else:
                    btc = BTC(targets, clf_predictions)
                    bec, conflicts = BEC(
                        targets, clf_predictions, track_conflict, True)
                    #print("BTC: ", btc)
                    #print("Instance BEC: ", bec)
                    grid_conflicts.append(conflicts)
                print(str().join(("Scoring error is: ",
                      str(scores['scorings'][i][j]))))
            except Exception as e:
                print("Something happened! Continuing...", logger.exception(e), test_scores['scorings'])
                if len(train_scores['scorings'][i])-1 == j:
                    train_scores['scorings'][i].pop(j)
                if len(train_scores['scorings'][i])-1 != j:
                    train_scores['scorings'][i].append(2)
                if len(test_scores['scorings'][i])-1 == j:
                    test_scores['scorings'][i].pop(j)
                if len(test_scores['scorings'][i])-1 != j:
                    test_scores['scorings'][i].append(2)
                if len(scores['scorings'][i])-1 == j:
                    scores['scorings'][i].pop(j)
                if len(scores['scorings'][i])-1 != j:
                    scores['scorings'][i].append(2)
                if len(scores['demographic_parity'][i])-1 == j:
                    scores['scorings'][i].pop(j)
                if len(scores['demographic_parity'][i])-1 != j:
                    scores['scorings'][i].append(100)
                if len(train_scores['demographic_parity'][i])-1 != j:
                    train_scores['demographic_parity'][i].append(2)
                if len(train_scores['demographic_parity'][i])-1 == j:
                    train_scores['demographic_parity'][i].pop(j)
                if len(test_scores['demographic_parity'][i])-1 != j:
                    test_scores['demographic_parity'][i].append(2)
                if len(test_scores['demographic_parity'][i])-1 == j:
                    test_scores['demographic_parity'][i].pop(j)
                models['models'][i].append(None)
                grid_conflicts.append(
                    np.array([i for i in range(len(features))]))
                continue    
        try:
            best_estimator['score'][i].append(min(scores['scorings'][i]))
            print('Best scores', best_estimator['score'], 'Scorings', scores['scorings'])
            best_model = models['models'][i][np.argmin(scores['scorings'][i])]
            best_estimator['C'][i].append(
                params['C'][0][np.array(scores['scorings'][i]).argmin()])
            best_estimator['reg_param'][i].append(
                params['reg_param'][0][np.array(scores['scorings'][i]).argmin()])
            print(str().join(("Best estimators are: ", "C: ", str(
                best_estimator['C'][i]), " Regularization parameter: ", str(best_estimator['reg_param'][i]))))
            print(str().join(("Mean training error scorings: ",
                  str(np.mean(train_scores['scorings'][i])))))
            print(str().join(("Mean test error scorings: ",
                  str(np.mean(test_scores['scorings'][i])))))
            print(str().join(("Mean training demographic parity scorings: ", str(
                np.mean(train_scores['demographic_parity'][i])))))
            print(str().join(("Mean test demographic parity scorings: ",
                  str(np.mean(test_scores['demographic_parity'][i])))))            
            print(str().join(("Standard dev train error scorings: ",
                  str(np.std(train_scores['scorings'][i])))))
            print(str().join(("Standard dev test error scorings: ",
                  str(np.std(test_scores['scorings'][i])))))
            print(str().join(("Mean error scorings: ",
                  str(np.mean(scores['scorings'][i])))))
            print(str().join(("Standard dev error scorings: ",
                  str(np.std(scores['scorings'][i])))))
            print(str().join(("Mean demographic parity scorings: ",
                  str(np.mean(scores['demographic_parity'][i])))))
        except Exception as e:
            logger.exception(e)
            return 'DiscriminationError'
    try:
        return best_estimator, best_model, grid_conflicts[np.argmin(scores['scorings'][i])]
    except Exception as e:
        return best_estimator, best_model, grid_conflicts
     
