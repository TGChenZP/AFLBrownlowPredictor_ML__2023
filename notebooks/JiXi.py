# 21/02/2023

import pandas as pd
import copy
import time
import numpy as np
import random
import pickle

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score





class JiXi:



    def __init__(self):
        """ Initialise class """
        self._initialise_objects()

        print('JiXi Initialised')



    def _initialise_objects(self):
        """ Helper to initialise objects """

        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.tuning_result = None
        self.model = None
        self.parameter_choices = None
        self.hyperparameters = None
        self.feature_n_ningxiang_score_dict = None
        self.non_tuneable_parameter_choices = list()
        self.checked = None
        self.result = None
        self.tuning_result_saving_address = None
        self.object_saving_address = None
        self._up_to = 0
        self._seed = 19421221
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self.combos = None
        self.n_items = None
        self.outmost_layer = None
        self._core = None
        self._relative_combos = None
        self._both_combos = None
        self._dealt_with = None
        self._pos_neg_combos = None
        self._abs_max = None
        self._new_combos = None
        self._parameter_value_map_index = None
        self._total_combos = None
        self._tune_features = None

        self.regression_extra_output_columns = ['Train r2', 'Val r2', 'Test r2', 
            'Train RMSE', 'Val RMSE', 'Test RMSE', 'Train MAPE', 'Val MAPE', 'Test MAPE', 'Time']
        self.classification_extra_output_columns = ['Train accu', 'Val accu', 'Test accu', 
            'Train balanced_accu', 'Val balanced_accu', 'Test balanced_accu', 'Train f1', 'Val f1', 'Test f1', 
            'Train precision', 'Val precision', 'Test precision', 'Train recall', 'Val recall', 'Test recall', 'Time']

        

    def read_in_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        """ Reads in train validate test data for tuning """

        self.train_x = train_x
        print("Read in Train X data")

        self.train_y = train_y
        print("Read in Train y data")

        self.val_x = val_x
        print("Read in Val X data")

        self.val_y = val_y
        print("Read in Val y data")

        self.test_x = test_x
        print("Read in Test X data")

        self.test_y = test_y
        print("Read in Test y data")



    def read_in_model(self, model, type):
        """ Reads in underlying model object for tuning, and also read in what type of model it is """

        assert type == 'Classification' or type == 'Regression' # check

        # record
        self.model = model
        self.clf_type = type 

        print(f'Successfully read in model {self.model}, which is a {self.clf_type} model')



    def set_hyperparameters(self, parameter_choices):
        """ Input hyperparameter choices """

        self.parameter_choices = parameter_choices
        self._sort_hyperparameter_choices()

        self.hyperparameters = list(parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [len(parameter_choices[key]) for key in self.hyperparameters]
        self._total_combos = np.prod(self.n_items)

        # automatically calculate all combinations and setup checked and result arrays and tuning result dataframe
        self._get_combinations()
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print("Successfully recorded hyperparameter choices")



    def _sort_hyperparameter_choices(self):
        """ Helper to ensure all hyperparameter choice values are in order from lowest to highest """

        for key in self.parameter_choices:
            tmp = copy.deepcopy(list(self.parameter_choices[key]))
            tmp.sort()
            self.parameter_choices[key] = tuple(tmp)



    def _get_combinations(self):
        """ Helper to calculate all combinations """

        ##ALGORITHM

        # recursively append values to get every combination in ordinal/numerical form
        self.combos = [[]]
        for i in range(len(self.n_items)):

            tmp = copy.deepcopy(self.combos)
            self.combos = list()

            for x in tmp:

                for k in range(self.n_items[i]):
                    y = copy.deepcopy(x)
                    
                    y.append(k)

                    self.combos.append(y)



    def _get_checked_and_result_array(self):
        """ Helper to set up checked and result array """

        self.checked = np.zeros(shape=self.n_items)
        self.result = np.zeros(shape=self.n_items)



    def _setup_tuning_result_df(self):
        """ Helper to set up tuning result dataframe """

        tune_result_columns = copy.deepcopy(self.hyperparameters)

        if self._tune_features:
            tune_result_columns.append('feature combo ningxiang score')

        # Different set of metric columns for different types of models
        if self.clf_type == 'Classification':
            tune_result_columns.extend(self.classification_extra_output_columns)
        elif self.clf_type == 'Regression':
            tune_result_columns.extend(self.regression_extra_output_columns)

        self.tuning_result = pd.DataFrame({col:list() for col in tune_result_columns})



    def set_non_tuneable_hyperparameters(self, non_tuneable_hyperparameter_choice):
        """ Input Non tuneable hyperparameter choice """

        if type(non_tuneable_hyperparameter_choice) is not dict:
            print('non_tuneable_hyeprparameters_choice must be dict, please try again')
            return
        
        for nthp in non_tuneable_hyperparameter_choice:
            if type(non_tuneable_hyperparameter_choice[nthp]) in (set, list, tuple, dict):
                print('non_tuneable_hyperparameters_choice must not be of array-like type')
                return

        self.non_tuneable_parameter_choices = non_tuneable_hyperparameter_choice

        print("Successfully recorded non_tuneable_hyperparameter choices")
    


    def set_features(self, ningxiang_output):
        """ Input features """

        if type(ningxiang_output) is not dict:
            print("Please ensure NingXiang output is a dict")
            return
        
        if not self.hyperparameters:
            print("Missing hyperparameter choices, please run .set_hyperparameters() first")
            return
        
        for feature in list(ningxiang_output.keys())[-1]:
            if feature not in list(self.train_x.columns):
                print(f'feature {feature} in ningxiang output is not in train_x. Please try again')
                return
            if feature not in list(self.val_x.columns):
                print(f'feature {feature} in ningxiang output is not in val_x. Please try again')
                return
            if feature not in list(self.test_x.columns):
                print(f'feature {feature} in ningxiang output is not in test_x. Please try again')
                return
        
        # sort ningxiang just for safety, and store up
        ningxiang_output_sorted = self._sort_features(ningxiang_output)
        self.feature_n_ningxiang_score_dict = ningxiang_output_sorted

        # activate this switch
        self._tune_features = True

        # update previous internal structures based on first set of hyperparameter choices
        ##here used numbers instead of tuples as the values in parameter_choices; thus need another mapping to get map back to the features
        self.parameter_choices['features'] = tuple([i for i in range(len(ningxiang_output_sorted))])
        self.feature_combo_n_index_map = {i: list(ningxiang_output_sorted.keys())[i] for i in range(len(ningxiang_output_sorted))}

        self.hyperparameters = list(self.parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [len(self.parameter_choices[key]) for key in self.hyperparameters]
        self._total_combos = np.prod(self.n_items)

        # automatically calculate all combinations and setup checked and result arrays and tuning result dataframe
        self._get_combinations()
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print("Successfully recorded tuneable feature combination choices and updated relevant internal structures")


    
    def _sort_features(self, ningxiang_output):
        """ Helper for sorting features based on NingXiang values (input dict output dict) """

        ningxiang_output_list = [(key, ningxiang_output[key]) for key in ningxiang_output]

        ningxiang_output_list.sort(key = lambda x:x[1])

        ningxiang_output_sorted = {x[0]:x[1] for x in ningxiang_output_list}

        return ningxiang_output_sorted

    

    def change_tuning_style(self, type, seed = None, outer_most_layer = 2, randomise = True): 
        """ Function which determines how to order the combinations for tuning """

        if not self.combos:
            print("Missing hyperparameter choices, please run .set_hyperparameters() first")
            return

        self.combos.sort() # to ensure functionality of seed, always sort first

        if type == 'a': # Sorted order (nested loops)
            # sorting operation conducted previously
            print('Changed tuning order to sorted')
        
        elif type == 'b': # Random order
            
            if seed:
                random.seed(seed)
            else:
                random.seed(self._seed)
            
            random.shuffle(self.combos)
            print('Changed tuning_order to randomised')
        
        elif type == 'c': # Layer by Layer 
            self._change_tuning_style_c(outer_most_layer, randomise, seed)
            if randomise:
                print(f'Changed tuning order to "Layer by Layer": {self.outmost_layer} Layers, randomised')
            else:
                print(f'Changed tuning order to "Layer by Layer": {self.outmost_layer} Layers, not randomised')

        elif type == 'd': # Diagonal + Horizontal first, before conducting layer by layer
            self._change_tuning_style_d()
            print(f'Changed tuning order to Diag-Hor -> "Layer by Layer": {self.outmost_layer} Layers, randomised')



    def _get_core(self):
        """ Helper to calculate core """
        self._core = [int(i/2) for i in self.n_items]



    def _get_relative_combos(self):
        """ Helper to calculate relative coordinates of combinations""" 
        self._relative_combos = [[combo[j] - self._core[j] for j in range(len(self.n_items))] for combo in self.combos]



    ### TYPE C
    def _get_both_combos(self):
        """ Helper to put (combos, relative combos) together into a tuple """
        self._get_relative_combos()
        self._both_combos = [[self.combos[i], self._relative_combos[i]] for i in range(len(self.combos))]



    def _get_layer_by_layer(self):
        """ Helper to get Layer by Layer order """

        retain = copy.deepcopy(self._both_combos)
        self._dealt_with = list()

        # starting with the outmost layer, ending with -1 because of the >
        for i in range(self.outmost_layer, -2, -1):
            tmp_retain = list()
            tmp_dealt_with = list()
            
            for item in retain:

                trigger = 1
                for j in item[1]:
                    if abs(j) > i:
                        tmp_dealt_with.append(item[0])
                        trigger = 0
                        break
                
                if trigger:
                    tmp_retain.append(item)

            retain = tmp_retain
            self._dealt_with.append(tmp_dealt_with)



    def _get_c_combos(self):
        """ Helper to get the combinations into Layer by Layer order """

        self.combos = list()
        # working with dealt_with backwards because we want the innermost layers first
        for i in range(len(self._dealt_with)-1, -1, -1):
            self.combos.extend(self._dealt_with[i])
            


    def _change_tuning_style_c(self, outmost_layer, randomise, seed):
        """ Helper to run all type-c helpers to get combinations into Layer by Layer order """

        if randomise:
            if seed:
                random.seed(seed)
            else:
                random.seed(self._seed)
        
        random.shuffle(self.combos)

        self.outmost_layer = outmost_layer
        self._get_core()
        self._get_both_combos()
        self._get_layer_by_layer()
        self._get_c_combos()



    ### TYPE D    
    def _change_tuning_style_d(self):
        """ Helper to all type-d helpers to get combinations into Layer by Layer order """

        self._change_tuning_style_c(outmost_layer= max(self.n_items), randomise = True, seed = self._seed)

        self._get_hor_combos()
        self._get_diag_combos()

        self._get_d_combos()



    def _get_hor_combos(self):
        """ Helper to get all combinations that lie on horizontal line from core """
        self._hor_combos = list()
        for i in range(len(self.n_items)):
            for j in range(self.n_items[i]):
                tmp = copy.deepcopy(self._core)
                tmp[i] = j
                self._hor_combos.append(tmp)



    def _get_pos_neg_combos(self):
        """ Helper to get all combinations of -1 and 1 """

        ##ALGORITHM
        self._pos_neg_combos = [[]]
        for i in range(len(self.n_items)):

            tmp = copy.deepcopy(self._pos_neg_combos)
            self._pos_neg_combos = list()

            for x in tmp:

                for k in (-1, 1):
                    y = copy.deepcopy(x)
                    
                    y.append(k)

                    self._pos_neg_combos.append(y)



    def _get_abs_max(self):
        """ Helper to get maximum absolute value of the relative combos """

        max_pos = max([self.n_items[i] - self._core[i] for i in range(len(self.n_items))])
        min_neg = min([0-self._core[i] for i in range(len(self.n_items))])

        self._abs_max = max((max_pos, abs(min_neg)))



    def _get_diag_combos(self):
        """ Helper to get all combinations that lie on diagonal line from core """

        # Implementation idea: first get all the diagonal combos (even those that go outside the field-space) 
        # by using all combinations of (-1, 1) multipled by each value from 1 to _abs_max, before eliminating combos that go
        # outside the field-space
        self._get_pos_neg_combos()
        self._get_abs_max()

        diag_rel_combos = list()
        for i in range(self._abs_max):
            diag_rel_combos.extend([[(i+1)*pos_neg_combo[j] for j in range(len(self.n_items))] for pos_neg_combo in self._pos_neg_combos])
        
        tmp_diag_combos = [[combo[j] + self._core[j] for j in range(len(self.n_items))] for combo in diag_rel_combos]

        self._diag_combos = list()
        
        for combo in tmp_diag_combos:
            trigger = 1
            for i in range(len(combo)):
                if combo[i] < 0 or combo[i] >= self.n_items[i]: # if outside field space then eliminate
                    trigger = 0
                    break
            
            if trigger:
                self._diag_combos.append(combo)



    def _get_d_combos(self):
        """ Helper to run all type-c helpers to get combinations into Diag-Hor -> Layer by Layer order """

        self._new_combos = list()

        self._new_combos.append(self._core)

        # put in diagonal first to get more variety
        for combo in self._diag_combos:
            if combo not in self._new_combos:
                self._new_combos.append(combo)

        for combo in self._hor_combos:
            if combo not in self._new_combos:
                self._new_combos.append(combo)

        # put in rest of the combos - already sorted in layer by layer order
        for combo in self.combos:
            if combo not in self._new_combos:
                self._new_combos.append(combo)
        
        self.combos = self._new_combos


        
    def tune(self, key_stats_only = False):
        """ Begin tuning """

        if self.train_x is None or self.train_y is None or self.val_x is None or self.val_y is None or self.test_x is None or self.test_y is None:
            print(" Missing one of the datasets, please run .read_in_data() ")
            return

        if self.model is None:
            print(" Missing model, please run .read_in_model() ")
            return
        
        if self.combos is None:
            print("Missing hyperparameter choices, please run .set_hyperparameters() first")
            return

        if self.tuning_result_saving_address is None:
            print("Missing tuning result csv saving address, please run ._save_tuning_result() first")

        self.key_stats_only = key_stats_only

        for combo in self.combos:

            if not self.checked[tuple(combo)]:
                
                self._up_to += 1
                self._train_and_test_combo(combo)
            
            else:
                self._check_already_trained_best_score(combo)

        # Display final information
        print("TUNING FINISHED\n")

        print('Max Score: \n', self.best_score)
        print('Max Combo: \n', self.best_combo)

        print('% Combos Checked:', int(sum(self.checked.reshape((np.prod(self.n_items))))), 'out of', np.prod(self.n_items), 'which is', f'{np.mean(self.checked).round(8)*100}%')



    def tune_parallel(self, part, splits, key_stats_only = False):
        """ Begin tuning in Parallel """

        assert type(part) is int and type(splits) is int

        if part <= 0 or part > splits:
            print("Part must be within [1, splits]")
            return

        if self.train_x is None or self.train_y is None or self.val_x is None or self.val_y is None or self.test_x is None or self.test_y is None:
            print(" Missing one of the datasets, please run .read_in_data() ")
            return

        if self.model is None:
            print(" Missing model, please run .read_in_model() ")
            return
        
        if self.combos is None:
            print("Missing hyperparameter choices, please run .set_hyperparameters() first")
            return

        if self.tuning_result_saving_address is None:
            print("Missing tuning result csv saving address, please run ._save_tuning_result() first")

        self.key_stats_only = key_stats_only

        parallel_combo_to_tune = copy.deepcopy(self.combos)
        start_index = int((part-1)/splits * len(self.combo))
        end_index = int(part/splits * len(self.combo))
        parallel_combo_to_tune = parallel_combo_to_tune[start_index:end_index]

        print(f'Parallel tuning part {part}: set to tune {len(parallel_combo_to_tune)} combinations')

        for combo in self.parallel_combo_to_tune:

            if not self.checked[tuple(combo)]:
                
                self._up_to += 1
                self._train_and_test_combo(combo)
            
            else:
                self._check_already_trained_best_score(combo)

        # Display final information
        print(f"PARALLEL TUNING PART {part} FINISHED\n")

        print('Max Score: \n', self.best_score)
        print('Max Combo: \n', self.best_combo)

        print('% Combos Checked:', int(sum(self.checked.reshape((np.prod(self.n_items))))), 'out of', np.prod(self.n_items), 'which is', f'{np.mean(self.checked).round(8)*100}%')



    def _train_and_test_combo(self, combo):
        """ Helper to train and test each combination as part of tune() """

        combo = tuple(combo)
        
        params = {self.hyperparameters[i]:self.parameter_choices[self.hyperparameters[i]][combo[i]] for i in range(len(self.hyperparameters))}
        
        
        if self._tune_features == True:
            del params['features']
            tmp_train_x = self.train_x[list(self.feature_combo_n_index_map[combo[-1]])] 
            tmp_val_x = self.val_x[list(self.feature_combo_n_index_map[combo[-1]])]
            tmp_test_x = self.test_x[list(self.feature_combo_n_index_map[combo[-1]])]

            # add non tuneable parameters
            for nthp in self.non_tuneable_parameter_choices:
                params[nthp] = self.non_tuneable_parameter_choices[nthp]

            # initialise object
            clf = self.model(**params)

            params['features'] = [list(self.feature_combo_n_index_map[combo[-1]])] 
            params['feature combo ningxiang score'] = self.feature_n_ningxiang_score_dict[self.feature_combo_n_index_map[combo[-1]]]

        else:
            tmp_train_x = self.train_x
            tmp_val_x = self.val_x
            tmp_test_x = self.test_x

            # add non tuneable parameters
            for nthp in self.non_tuneable_parameter_choices:
                params[nthp] = self.non_tuneable_parameter_choices[nthp]

            # initialise object
            clf = self.model(**params)

        # get time and fit
        start = time.time()
        clf.fit(tmp_train_x, self.train_y)
        end = time.time()

        # get predicted labels/values for three datasets
        train_pred = clf.predict(tmp_train_x)
        val_pred = clf.predict(tmp_val_x)
        test_pred = clf.predict(tmp_test_x)

        # get scores and time used
        time_used = end-start

        # build output dictionary and save result
        df_building_dict = params


        if self.clf_type == 'Regression':

            train_score = val_score = test_score = train_rmse = val_rmse = test_rmse = train_mape = val_mape = test_mape = 0

            try:
                train_score = r2_score(self.train_y, train_pred)
            except:
                pass
            try:
                val_score = r2_score(self.val_y, val_pred)
            except:
                pass
            try:
                test_score = r2_score(self.test_y, test_pred)
            except:
                pass
            
            try:
                train_rmse = np.sqrt(mean_squared_error(self.train_y, train_pred))
            except:
                pass
            try:
                val_rmse = np.sqrt(mean_squared_error(self.val_y, val_pred))
            except:
                pass
            try:
                test_rmse = np.sqrt(mean_squared_error(self.test_y, test_pred))
            except:
                pass

            if self.key_stats_only == False:
                try:
                    train_mape = mean_absolute_percentage_error(self.train_y, train_pred)
                except:
                    pass
                try:
                    val_mape = mean_absolute_percentage_error(self.val_y, val_pred)
                except:
                    pass
                try:
                    test_mape = mean_absolute_percentage_error(self.test_y, test_pred)
                except:
                    pass
            
            df_building_dict['Train r2'] = [np.round(train_score, 6)]
            df_building_dict['Val r2'] = [np.round(val_score, 6)]
            df_building_dict['Test r2'] = [np.round(test_score, 6)]
            df_building_dict['Train RMSE'] = [np.round(train_rmse, 6)]
            df_building_dict['Val RMSE'] = [np.round(val_rmse, 6)]
            df_building_dict['Test RMSE'] = [np.round(test_rmse, 6)]
            
            if self.key_stats_only == False:
                df_building_dict['Train MAPE'] = [np.round(train_mape, 6)]
                df_building_dict['Val MAPE'] = [np.round(val_mape, 6)]
                df_building_dict['Test MAPE'] = [np.round(test_mape, 6)]

        
        elif self.clf_type == 'Classification':

            train_score = val_score = test_score = train_bal_accu = val_bal_accu = test_bal_accu = train_f1 = val_f1 = test_f1 = \
                train_precision = val_precision = test_precision = train_recall = val_recall = test_recall = 0

            try:    
                train_score = accuracy_score(self.train_y, train_pred)
            except:
                pass
            try:
                val_score = accuracy_score(self.val_y, val_pred)
            except:
                pass
            try:
                test_score = accuracy_score(self.test_y, test_pred)
            except:
                pass

            try:
                train_bal_accu = balanced_accuracy_score(self.train_y, train_pred)
            except:
                pass
            try:
                val_bal_accu = balanced_accuracy_score(self.val_y, val_pred)
            except:
                pass
            try:
                test_bal_accu = balanced_accuracy_score(self.test_y, test_pred)
            except:
                pass
            
            try:
                train_f1 = f1_score(self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                val_f1 = f1_score(self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                test_f1 = f1_score(self.test_y, test_pred, average='weighted')
            except:
                pass
            
            try:
                train_precision = precision_score(self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                val_precision = precision_score(self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                test_precision = precision_score(self.test_y, test_pred, average='weighted')
            except:
                pass

            try:
                train_recall = recall_score(self.train_y, train_pred, average='weighted')
            except:
                pass
            try:
                val_recall = recall_score(self.val_y, val_pred, average='weighted')
            except:
                pass
            try:
                test_recall = recall_score(self.test_y, test_pred, average='weighted')
            except:
                pass

            df_building_dict['Train accu'] = [np.round(train_score, 6)]
            df_building_dict['Val accu'] = [np.round(val_score, 6)]
            df_building_dict['Test accu'] = [np.round(test_score, 6)]
            df_building_dict['Train balanced_accuracy'] = [np.round(train_bal_accu, 6)]
            df_building_dict['Val balanced_accuracy'] = [np.round(val_bal_accu, 6)]
            df_building_dict['Test balanced_accuracy'] = [np.round(test_bal_accu, 6)]
            df_building_dict['Train f1'] = [np.round(train_f1, 6)]
            df_building_dict['Val f1'] = [np.round(val_f1, 6)]
            df_building_dict['Test f1'] = [np.round(test_f1, 6)]
            df_building_dict['Train precision'] = [np.round(train_precision, 6)]
            df_building_dict['Val precision'] = [np.round(val_precision, 6)]
            df_building_dict['Test precision'] = [np.round(test_precision, 6)]
            df_building_dict['Train recall'] = [np.round(train_recall, 6)]
            df_building_dict['Val recall'] = [np.round(val_recall, 6)]
            df_building_dict['Test recall'] = [np.round(test_recall, 6)]


        df_building_dict['Time'] = [np.round(time_used, 2)]


        tmp = pd.DataFrame(df_building_dict)


        self.tuning_result = self.tuning_result.append(tmp)
        self._save_tuning_result()

        # update best score stats
        if val_score > self.best_score: 
            self.best_score = val_score
            self.best_clf = clf
            self.best_combo = combo

        # update internal governing DataFrames
        self.checked[combo] = 1
        self.result[combo] = val_score

        print(f'''Trained and Tested combination {self._up_to} of {self._total_combos}: {combo}, taking {np.round(time_used,2)} seconds to get val score of {np.round(val_score,4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}''')



    def _check_already_trained_best_score(self, combo):
        """ Helper for checking whether an already trained combo is best score """
        
        # update best score stats
        if self.result[combo] > self.best_score: 
            self.best_score = self.result[combo]
            self.best_clf = None
            print(f"As new Best Combo {combo} was read in, best_clf is set to None")
            self.best_combo = combo

        print(f'''Already Trained and Tested combination {combo}, which had val score of {np.round(self.result[combo],4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}. 
        Has trained {self._up_to} of {self._total_combos} combinations so far''')



    def _save_tuning_result(self):
        """ Helper to export tuning result csv """

        tuning_result_saving_address_strip = self.tuning_result_saving_address.split('.csv')[0]

        self.tuning_result.to_csv(f'{tuning_result_saving_address_strip}.csv', index=False)


    
    def view_best_combo_and_score(self):
        """ View best combination and its validation score """
        
        print(f'(Current) Best combo: {self.best_combo} with val score {self.best_score}')

    

    def read_in_tuning_result_df(self, address): 
        """ Read in tuning result csv and read data into checked and result arrays """

        if self.parameter_choices is None:
            print("Missing parameter_choices to build _parameter_value_map_index, please run set_hyperparameters() first")

        if self.clf_type is None:
            print('Missing clf_type. Please run .read_in_model() first.')

        self.tuning_result = pd.read_csv(address)
        print(f"Successfully read in tuning result of {len(self.tuning_result)} rows")

        self._up_to = 0

        self._create__parameter_value_map_index()

        # read DataFrame data into internal governing DataFrames of JiXi
        for row in self.tuning_result.iterrows():

            self._up_to += 1

            combo = list()
            for hyperparam in self.hyperparameters:
                if hyperparam == 'features':
                    
                    # reverse two dicts
                    ningxiang_score_n_feature_dict = {self.feature_n_ningxiang_score_dict[key]:key for key in self.feature_n_ningxiang_score_dict}
                    index_n_feature_combo_map = {self.feature_combo_n_index_map[key]:key for key in self.feature_combo_n_index_map}
                    # special input
                    combo.append(index_n_feature_combo_map[ningxiang_score_n_feature_dict[row[1]['feature combo ningxiang score']]])
                    
                else:
                    combo.append(self._parameter_value_map_index[hyperparam][row[1][hyperparam]])
            
            self.checked[combo] = 1
            
            if self.clf_type == 'Regression':
                self.result[combo] = row[1]['Val r2']
            elif self.clf_type == 'Classification':
                self.result[combo] = row[1]['Val accu']


    
    def _create__parameter_value_map_index(self):
        """ Helper to create parameter-value index map """

        self._parameter_value_map_index = dict()
        for key in self.parameter_choices.keys():
            tmp = dict()
            for i in range(len(self.parameter_choices[key])):
                tmp[self.parameter_choices[key][i]] = i
            self._parameter_value_map_index[key] = tmp
    


    def set_tuning_result_saving_address(self, address):
        """ Read in where to save tuning object """

        self.tuning_result_saving_address = address
        print('Successfully set tuning output address')


    
    def _set_object_saving_address(self, address):
        """ Read in where to save the JiXi object """

        self.object_saving_address = address
        print('Successfully set object output address')



    def export_jixi(self, address):
        """ Export jixi object """

        self._set_object_saving_address(address)

        # copy object and set big objects to None
        object_save = copy.deepcopy(self)
        
        object_save.train_x = None
        object_save.train_y = None
        object_save.val_x = None
        object_save.val_y = None
        object_save.test_x = None
        object_save.test_y = None
        object_save._up_to = 0

        # Export
        object_saving_address_strip = self.object_saving_address.split('.pickle')[0]
        with open(f'{object_saving_address_strip}.pickle', 'wb') as f:
            pickle.dump(object_save, f)

        print(f'Successfully exported JiXi object as {self.object_saving_address}')