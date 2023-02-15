# 15/02/2023

import pandas as pd
import numpy as np
import statistics as s
import copy
import time
from itertools import combinations
import pickle
import random

from scipy.spatial.distance import cdist
from scipy.stats import t
from scipy import stats

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score





class YangZhou:

    def __init__(self):
        """ Initialise class """
        self._initialise_objects()

        print('YangZhou Initialised')



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
        self.checked = None
        self.result = None
        self.checked_core = None
        self.been_best = None
        self.been_cruised = None
        self.tuning_result_saving_address = None
        self.object_saving_address = None
        self._parameter_value_map_index = None
        self._seed = 19260817
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self.n_items = None
        self._core = None
        self._cruise_indices = None
        self._cruise_indices_values = None
        self._cruise_combinations = None
        self._restarts = 0
        self._cruising = True
        self._surrounding_vectors = None
        self._total_combos = None

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
        print("Read in Train x data")

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
        self.num_hyperparameters = {hyperparameter:len(parameter_choices[hyperparameter]) for hyperparameter in self.hyperparameters}
        self._total_combos = np.prod(self.n_items)

        # automatically setup checked and result arrays and tuning result dataframe
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print("Successfully recorded hyperparameter choices")


    
    def _sort_hyperparameter_choices(self):
        """ Helper to ensure all hyperparameter choice values are in order from lowest to highest """

        for key in self.parameter_choices:
            tmp = copy.deepcopy(list(self.parameter_choices[key]))
            tmp.sort()
            self.parameter_choices[key] = tuple(tmp)



    def _get_checked_and_result_array(self):
        """ Helper to set up checked and result array """

        self.checked = np.zeros(shape=self.n_items)
        self.result = np.zeros(shape=self.n_items)
        self.checked_core = np.zeros(shape=self.n_items)
        self.been_best = np.zeros(shape=self.n_items) # strictly for last part of Guidance Algorithm
        self.been_cruised = np.zeros(shape=self.n_items)



    def _setup_tuning_result_df(self):
        """ Helper to set up tuning result dataframe """

        tune_result_columns = copy.deepcopy(self.hyperparameters)

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
        
        if not self.combos:
            print("Missing hyperparameter choices, please run .set_hyperparameters() first")
            return
        
        # sort ningxiang just for safety, and store up
        ningxiang_output_sorted = self._sort_features(ningxiang_output)
        self.feature_n_ningxiang_score_dict = ningxiang_output_sorted

        # activate this switch
        self._tune_features = True

        # update previous internal structures based on first set of hyperparameter choices
        ##here used numbers instead of tuples as the values in parameter_choices; thus need another mapping to get map back to the features
        self.parameter_choices['features'] = tuple([i for i in range(len(ningxiang_output_sorted))])
        self.feature_combo_n_index_map = {i: ningxiang_output_sorted.keys()[i] for i in range(len(ningxiang_output_sorted))}

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



    def _get_core(self):
        """ Helper to calculate core """
        self._core = [int(i/2) for i in self.n_items]



    def _get_cruise_combinations(self):
        """ Helper to cruise combinations """

        self._get_cruise_indices_values()
        self._generate_cruise_combinations() # first get cruise indicies, then use indicies to get combinations



    def _get_cruise_indices_values(self):
        """ Helper to get cruise indices values of each dimension which serves as building blocks for cruise combinations """

        self._cruise_indices = dict()
        for hyperparameter in self.hyperparameters:
            self._cruise_indices[hyperparameter] = self._get_cruise_indices_1d(d_val = self.num_hyperparameters[hyperparameter], max_jump = 5)

        self._cruise_indices_values = list(self._cruise_indices.values())



    def _get_cruise_indices_1d(self, d_val, max_jump = 5): 
        """ Helper that returns the appropriate cruise indices based on the number of values in dimension. Second argument controls maximum split size, defaulted to 5 """

        assert type(d_val) is int and type(max_jump) is int, "Error: type of input(s) is not int"
        assert max_jump >= 1, "Error: max_jump must be >= 1"

        gap = d_val - 1
        split = ((gap-1)//max_jump)

        jump = self._find_gaps(split, gap)

        cruise_indices_1d = self._find_cruise_indices_1d(jump)

        return cruise_indices_1d



    def _find_gaps(self, split, gap):
        """ Helper that finds the size of jumps between each element of the final cruise indices, as evenly split as possible with jump size <= 5 """

        if split > 0:
            jump = [gap//(split+1) for i in range(split+1)]
            diff = gap - sum(jump)
            if diff:
                for i in range(diff):
                    jump[i] += 1
        else:
            jump = [gap]

        return jump



    def _find_cruise_indices_1d(self, jump):
        """ Helper that finds the actual cruise_indices based on gaps """

        cruise_indices_1d = [0]
        for i in range(len(jump)):
            cruise_indices_1d.append(sum(jump[:i+1]))

        if cruise_indices_1d == [0, 0]:
            cruise_indices_1d = [0]
            
        return cruise_indices_1d



    def _generate_cruise_combinations(self):
        """ Helper that generates the actual cruise combinations based on cruise indicies """
        ##ALGORITHM: how to generate all combinations of any dimensions given each dimension has different values
        self._cruise_combinations = [[]]
        for i in range(len(self._cruise_indices_values)):

            tmp = copy.deepcopy(self._cruise_combinations)
            self._cruise_combinations = list()

            for x in tmp:

                for k in self._cruise_indices_values[i]:
                    y = copy.deepcopy(x)
                    
                    y.append(k)

                    self._cruise_combinations.append(y)



    def _sort_cruise_combos(self, max_combo):
        """ sort the cruise combos based on Euclidean distance from current max"""

        edist = list(cdist([max_combo], self._cruise_combinations).flatten())
        ordered_cruise_combos = [(self._cruise_combinations[i], edist[i]) for i in range(len(self._cruise_combinations))]

        ordered_cruise_combos.sort(reverse=True, key=lambda x: x[1])

        sorted_cruise_combos = [ordered_cruise_combos[i][0] for i in range(len(ordered_cruise_combos))]

        return sorted_cruise_combos

    

    def _get_max_surrounding_mean_sd(self):
        """ Helper to get the surrounding mean and sd given the current maximum """

        best_combo_surrounding_combos = self._get_surrounding_combos(self.best_combo, self._surrounding_vectors)
        best_combo_surrounding_scores = [self.best_score]
        for combo in best_combo_surrounding_combos:
            best_combo_surrounding_scores.append(self.result[tuple(combo)])

        max_surrounding_mean = s.mean(best_combo_surrounding_scores)
        max_surrounding_sd = s.stdev(best_combo_surrounding_scores)

        return max_surrounding_mean, max_surrounding_sd



    def _cruise_warning_threshold(self, max_surrounding_mean, max_surrounding_sd, max_surrounding_n):
        """ Helper that gets the warning threshold by (mean of best combo surrounds - halfwidth) """

        # use 0.95 (one sided test)
        qt = t.ppf(0.95, max_surrounding_n-1) 
        halfwidth = max_surrounding_sd * qt * 1/np.sqrt(max_surrounding_n)

        return max_surrounding_mean - halfwidth



    def _CruiseSystem(self):
        """ Helper that performs cruising """

        print(f"Cruising: round {self._restarts}\n") 

        # get cruise combos in sorted order (furthest away from current max)
        sorted_cruise_combos = self._sort_cruise_combos(self.best_combo)

        # calculate warning threshold by getting max_surrounding_sd first
        max_surrounding_mean, max_surrounding_sd = self._get_max_surrounding_mean_sd()

        warning_threshold = self._cruise_warning_threshold(max_surrounding_mean, max_surrounding_sd, len(self._surrounding_vectors))

        # check each cruise combo
        for cruise_combo in sorted_cruise_combos:

            # only search if it hasn't been cruised before (if has then is not an artifect of significance)
            if not self.been_cruised[tuple(cruise_combo)]:
                
                self.been_cruised[tuple(cruise_combo)] = 2 # actually been cruised

                # if above warning threshold, then stop cruise and restart guide
                if self.result[tuple(cruise_combo)] >= warning_threshold:
                   
                    print(f"Cruise suspended due to suspicious case")
                    
                    return cruise_combo


        # if reach here then all cruise indicies checked. can safely say end cruise
        self._cruising = False

        return



    def _get_core(self):
        """ Helper to calculate core """

        self._core = [int(i/2) for i in self.n_items]



    def _get_surrounding_vectors(self, core):
        """ Helper that gets the VECTORS that moves the core to the COORDINATES that form the 3^d object around it """
        ##ALGORITHM: how to generate all combinations of any dimensions given each dimension has different values
        values = [-1, 0, 1]
        new_surroundings = [[-1], [0], [1]]

        for i in range(len(core) - 1):
            old_surroundings = copy.deepcopy(new_surroundings)
            new_surroundings = list()

            for surrounding in old_surroundings:
                for value in values:
                    new_surroundings.append(
                        [surrounding[i] if i < len(surrounding) else value for i in range(len(surrounding) + 1)])

        return new_surroundings



    def _get_surrounding_combos(self, core, surrounding_vectors):
        """ Helper that uses surrounding VECTORS to find surrounding COORDINATES """
        
        # Note this surrounding vector is not the same as the static local object generated by _get_surrounding_vectors

        assert len(surrounding_vectors) > 0
        assert len(surrounding_vectors[0]) == len(core)

        # generate the surrounding combos (checking for whether they are still in the field) 
        surrounding_combos = list()
        for i in range(len(surrounding_vectors)):
            new_combo = self._new_combos(core, surrounding_vectors[i])
            if new_combo is not False:
                surrounding_combos.append(new_combo)

        return surrounding_combos



    def _new_combos(self, core, vector):
        """ Helper that gets particular COORDINATE using a move in direction of VECTOR from particular CORE """

        assert len(core) == len(vector)

        new_combo = list()
        for i in range(len(vector)):
            val = core[i] + vector[i]
            if val >= self.n_items[i] or val < 0: # check whether combo is still in the field
                return False
            new_combo.append(val)

        return new_combo



    def _find_horizontal(self, surrounding_combos, core):
        """ Helper that finds the treatment and nulls block from a 'Horizontal' vector move """

        treatment = list()
        null = list()
        direction = list()

        for i in range(len(core)):

            for move in [-1, 1]:
                treatment_target = core[i] + move
                null_target = core[i]

                treatment_tmp = list()
                null_tmp = list()

                for vector in surrounding_combos:
                    if vector[i] == treatment_target:
                        treatment_tmp.append(vector)
                    elif vector[i] == null_target:
                        null_tmp.append(vector)

                treatment.append(treatment_tmp)
                null.append(null_tmp)
                direction.append([move if j == i else 0 for j in range(len(core))])

        return treatment, null, direction



    def _pick_x(self, i, core):
        """ Helper that picks all combinations of range(len(core)) for indexing when getting diagonal treatments """

        return list(combinations(list(range(len(core))), i))



    def get_indices(self, core):
        """ Helper that gets combinations of index to be used to find diagonal treatments: part of special algorithm """

        indices = list()
        for i in range(len(core)):
            for obj in self._pick_x(i, core):
                indices.append(obj)

        return indices



    def _find_diagonal(self, core, indices):
        """ Helper that finds the treatment and nulls block from a 'Diagonal' vector move (effectively interaction of all params) """

        treatment = list()
        null = list()

        diagonals = self._get_diagonals()

        for diagonal in diagonals:
            treatment.append(self._get_diagonal_treatment(core, diagonal, indices))

            null.append(self._get_diagonal_null(core, diagonal))

        return treatment, null, diagonals



    def _get_diagonals(self):
        """ Helper that finds all the diagonal vectors """

        return [obj for obj in self._surrounding_vectors if 0 not in obj]



    def _get_diagonal_treatment(self, core, diagonal, indices):
        """ Helper that finds all diagonal treatments of this diagonal direction - any vector that has from 1 to d elements in same
        direction diagonal, and all other vector positions 0 """

        treatment = self._get_surrounding_combos(core, self._get_diag_treatment_vectors(indices, diagonal))

        return treatment



    def _get_diag_treatment_vectors(self, indices, diagonal):
        """ Helper that finds all vectors for diagonal treatments (orthogonal to direction or 0 vector) """

        diag_vectors = list()

        for index in indices:
            tmp = [0 for i in range(len(diagonal))]
            for i in index:
                tmp[i] = diagonal[i]
            diag_vectors.append(tmp)

        return diag_vectors



    def _get_diagonal_null(self, core, diagonal):
        """ Helper that finds all diagonal nulls of this diagonal direction - any vector that is orthogonal to the current direction """

        null = list()
        for surrounding_vector in self._surrounding_vectors:
            if np.dot(surrounding_vector, diagonal) == 0:
                new_combo = self._new_combos(core, surrounding_vector)
                if new_combo is not False:
                    null.append(new_combo)

        return null



    def _get_blocks(self, core, surrounding_combos, indices):
        """ Helper that gets all blocks' treatments and nulls (in respective lists) (both horizontal and diagonal) """

        treatment = list()
        null = list()
        direction = list()

        hor_treatment, hor_null, hor_dir = self._find_horizontal(surrounding_combos, core)

        diag_treatment, diag_null, vert_dir = self._find_diagonal(core, indices)

        treatment.extend(hor_treatment)
        treatment.extend(diag_treatment)

        null.extend(hor_null)
        null.extend(diag_null)

        direction.extend(hor_dir)
        direction.extend(vert_dir)

        return treatment, null, direction



    def _get_treat_or_null_tune_scores(self, treat_or_null):
        """ Helper that returns as the relevant scores as a list to be used for t_test """

        treat_or_null_score = dict()

        for combo in treat_or_null:
            treat_or_null_score[tuple(combo)] = self.result[tuple(combo)]

        return treat_or_null_score



    def _dict_arg_max(self, dic):
        """ Helper that finds key of maximum dict value """
        
        max_val = -np.inf
        for key in dic:
            if dic[key] > max_val:
                max_val = dic[key]
                max_key = key

        return max_key



    def _xlnx(self, x):
        """ Helper that returns x*ln(x), rounding up to next int """
        y = int(x*np.log(x))+1
        
        return y



    def _find_new_core(self, treatment, null, direction, core):
        """ Helper that finds new cores - only first d*ln(d) treatments (actual combo representing direction) with positive mean difference and p value < 0.05 """

        assert len(treatment) == len(null)
        assert len(treatment) == len(direction)

        new_cores = []

        tmp_new_cores = list()
        
        for i in range(len(treatment)): 
            if len(treatment[i]) <= 1 or len(null[i]) <= 1:
                continue

            bool_inc = np.mean(list(self._get_treat_or_null_tune_scores(treatment[i]).values())) > np.mean(
                list(self._get_treat_or_null_tune_scores(null[i]).values()))
            
            if bool_inc: # check if the treatment mean > null mean

                p_val = stats.ttest_ind(list(self._get_treat_or_null_tune_scores(treatment[i]).values()),
                                    list(self._get_treat_or_null_tune_scores(null[i]).values()),
                                    equal_var=False).pvalue

                if p_val < 0.05: # check if p value < 0.05
                    
                    # get the main combo of treatment = direction + core as a proposed new core
                    proposed_new_core = tuple([direction[i][j] + core[j] for j in range(len(core))])

                    # get the max combo of the treatment
                    max_combo_of_treatment = self._dict_arg_max(self._get_treat_or_null_tune_scores(treatment[i])) 

                    # test if proposed new core is still on the grid
                    add_dir = True
                    for j in range(len(proposed_new_core)):
                        if proposed_new_core[j] < 0 or proposed_new_core[j] >= self.n_items[j]:
                            add_dir = False
                            break
                    
                    # if proposed new core is on the grid, then put it in tmp_new_core along with its p value
                    if add_dir:
                        if not self.checked_core[ proposed_new_core ]:
                            tmp_new_cores.append([proposed_new_core, p_val])

                    # if not, put in the max core
                    else:
                        if not self.checked_core[ tuple(max_combo_of_treatment) ]:
                            tmp_new_cores.append([tuple(max_combo_of_treatment), p_val])
                        
        # sort the tmp new cores list according to p values
        tmp_new_cores.sort(key = lambda x:x[1])

        # calculate how many new cores to accept according to the dimension of the grid (x = dim; accept x*ln(x))
        n_accept = self._xlnx(len(core))
        
        for i in range(len(tmp_new_cores)):
            if i >= n_accept:
                break

            new_cores.append(tmp_new_cores[i][0])
            self.checked_core[ tmp_new_cores[i][0] ] = 1

        return new_cores



    def _get_new_cores(self, core):
        """ Helper that gets new cores given old cores (performs the welch tests) """

        # if (should be rare) case where core has been a core before, then skip. For prevention of infinite loops
        # 2 means actual checked core, 1 means appended to checked core list but not checked
        if self.checked_core[tuple(core)] == 2:
            return
        else:
            self.checked_core[tuple(core)] = 2 # actually been checked as a core
            

        # prepare data for welch test
        surrounding_combos = self._get_surrounding_combos(core, self._surrounding_vectors)

        indices = self.get_indices(core)

        # put combos into treatments and nulls
        treatment, null, direction = self._get_blocks(core, surrounding_combos, indices)

        # actually tune the surrounding combos
        for combo in surrounding_combos:
            
            if self.checked[tuple(combo)] == 0:
                self._up_to += 1
                self._train_and_test_combo(combo)
            else:
                print(f'\tAlready Trained and Tested combination {combo}')

        # perform welch test and return surrounding combos that should be used as new core
        new_cores = self._find_new_core(treatment, null, direction, core)

        return new_cores  



    def _GuidanceSystem(self, core):
        """ Helper that performs guidance search """

        if self._restarts == 0:
            print("Guidance: initial round \n")
        else:
            print("Guidance: round", self._restarts, '\n')

        print('\tround', self._restarts, 'iteration: ', 0, '\n')

        # first get a surrounding 3^d tuned
        new_cores = self._get_new_cores(core)
        self.been_cruised[tuple(core)] = 1 # represent don't need to be added as a to cruised - but not cruised, rather a core

        round = 1
        while new_cores: # while new cores are being added
            print('\tround', self._restarts, "iteration: ", round, "\n") 
            round += 1

            old_new_cores = copy.deepcopy(new_cores)
            new_cores = list()

            # for each of the new cores, 'recursively' tune and grab new cores; but each Iteration doesn't end until all cores of current round has been checked
            for new_core in old_new_cores:
                
                new_new_cores = self._get_new_cores(new_core)

                if not new_new_cores:
                    new_new_cores = []

                self.been_cruised[tuple(core)] == 1
                
                for new_new_core in new_new_cores:
                    if self.checked_core[tuple(new_new_core)] == 0:
                        new_cores.append(new_new_core)
                        self.checked_core[tuple(new_new_core)] = 1 # represent added to checked core - to prevent repeated added to core


        # for current max, get 3^d block. if new max happens to be found, continue to do 3^d block until no new max is found
        # just a cheap way to flesh out the max (the goal of YangZhou)
        while self.been_best[tuple(self.best_combo)] == 0:

            self.been_best[tuple(self.best_combo)] = 1 
    
            surrounding_combos = self._get_surrounding_combos(self.best_combo, self._surrounding_vectors)
            for combo in surrounding_combos:
                
                if self.checked[tuple(combo)] == 0:
                    self._up_to += 1
                    self._train_and_test_combo(combo)
                else:
                    print(f'\tAlready Trained and Tested combination {combo}')

        # print information of this round 

        print('% Combos Checked Thus Far:', int(sum(self.checked.reshape((np.prod(self.n_items))))), 'out of', np.prod(self.n_items), 'which is', f'{np.mean(self.checked).round(8)*100}%')



    def tune(self, key_stats_only = False):
        """ Begin tuning """

        if self.train_x is None or self.train_y is None or self.val_x is None or self.val_y is None or self.test_x is None or self.test_y is None:
            print(" Missing one of the datasets, please run .read_in_data() ")
            return

        if self.model is None:
            print(" Missing model, please run .read_in_model() ")
            return

        if self.tuning_result_saving_address is None:
            print("Missing tuning result csv saving address, please run ._save_tuning_result() first")

        self.key_stats_only = key_stats_only

        print("BEGIN TUNING\n\n") 
        
        # FIRST: get all cruise combinations as well as core, and tune all these
        self._get_core()
        self._get_cruise_combinations() 

        first_round_combinations = copy.deepcopy(self._cruise_combinations)
        first_round_combinations.append(self._core) 

        random.seed(self._seed)
        random.shuffle(first_round_combinations)

        print("STAGE ZERO: Tune all Cruise combinations\n\n")
        for combo in first_round_combinations:
            
            if not self.checked[tuple(combo)]:
                self._up_to += 1
                self._train_and_test_combo(combo)
            
            else:
                print(f'\tAlready Trained and Tested combination {combo}')
        

        # SECOND: from the core combo, begin guidance system
        self._surrounding_vectors = self._get_surrounding_vectors(self._core)

        print('\n')
        print("STAGE ONE: Begin initial Guidance system\n\n")

        self._restarts = 0
        self._GuidanceSystem(self._core) # Initial Round of Guidance, starting at the core
        self._restarts += 1

        # THIRD: Recursively Cruise and restart Guide if find a combo that is within halfwidth of mean of best combo surrounds
        print("STAGE TWO: Begin Cruise system\n\n")
        self._cruising = True
        while self._cruising:
            suspicious_case_combo = self._CruiseSystem()

            if self._cruising:
                self._GuidanceSystem(tuple(suspicious_case_combo))
                self._restarts += 1

        # FINALLY: Final extensive guidance search around maxes.
        print("FINAL STAGE: Begin final Guidance system\n\n")
        old_best_score = copy.deepcopy(self.best_score)
        self._restarts = 'FINAL'

        self._GuidanceSystem(self.best_combo)

        while(self.best_score-old_best_score > 0):
            old_best_score = copy.deepcopy(self.best_score)
            self._GuidanceSystem(self.best_combo)



        # Display final information
        print("TUNING FINISHED\n")

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
        for nthp in self.non_tuneable_parameter_choices:
            del params[nthp] 


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


        print(f'''Trained and Tested combination {self._up_to} of {self._total_combos}: {combo}, taking {time_used} seconds to get val score of {val_score}
        Current best combo: {self.best_combo} with val score {self.best_score}''')



    def _save_tuning_result(self):
        """ Helper to export tuning result csv """

        tuning_result_saving_address_split = self.tuning_result_saving_address.split('.csv')[0]

        self.tuning_result.to_csv(f'{tuning_result_saving_address_split}.csv', index=False)

    

    def view_best_combo_and_score(self):
        """ View best combination and its validation score """
        
        print(f'(Current) Best combo: {self.best_combo} with val score {self.best_score}')



    def read_in_tuning_result_df(self, address): 
        """ Read in tuning result csv and read data into checked and result arrays """

        if self.parameter_choices is None:
            print("Missing parameter_choices to build parameter_value_map_index, please run set_hyperparameters() first")

        if self.clf_type is None:
            print('Missing clf_type. Please run .read_in_model() first.')

        self.tuning_result = pd.read_csv(address)
        print(f"Successfully read in tuning result of {len(self.tuning_result)} rows")

        self._up_to = 0

        self._create_parameter_value_map_index()

        # read DataFrame data into internal governing DataFrames of YangZhou
        for row in self.tuning_result.iterrows():

            self._up_to += 1
    
            combo = tuple([self.parameter_value_map_index[hyperparam][row[1][hyperparam]] for hyperparam in self.hyperparameters])
            
            self.checked[combo] = 1
            
            if self.clf_type == 'Regression':
                self.result[combo] = row[1]['Val r2']
            elif self.clf_type == 'Classification':
                self.result[combo] = row[1]['Val accu']
        
            # update best score stats
            if self.result[combo] > self.best_score: 
                self.best_score = self.result[combo]
                self.best_clf = None
                print(f"As new Best Combo {combo} is read in, best_clf is set to None")
                self.best_combo = combo


    
    def _create_parameter_value_map_index(self):
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
        """ Read in where to save the YangZhou object """

        self.object_saving_address = address
        print('Successfully set object output address')



    def export_yangzhou(self, address):
        """ Export yangzhou object """

        self._set_object_saving_address(address)

        # copy object and set big objects to None
        object_save = copy.deepcopy(self)
        
        object_save.train_x = None
        object_save.train_y = None
        object_save.val_x = None
        object_save.val_y = None
        object_save.test_x = None
        object_save.test_y = None

        # Export
        object_saving_address_split = self.object_saving_address.split('.pickle')[0]

        with open(f'{object_saving_address_split}.pickle', 'wb') as f:
            pickle.dump(object_save, f)

        print(f'Successfully exported YangZhou object as {self.object_saving_address}')