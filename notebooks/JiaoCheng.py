from ZhongShan import *
import pickle
import copy
import numpy as np
import json





class JiaoCheng:

    def __init__(self, sanmin):
        """ Initialise class """
        self.sanmin = sanmin
        self._initialise_objects()

        print('JiaoCheng Initialised')



    def _initialise_objects(self):
        pass
    

    
    def get_feature_combinations(self, score_type, label, penalty_function_type, export_address = None):
        """ Function that gets combinations based on JiaoCheng's algorithm along with its score, based on
        inputted score type, label and penalty function type. Has option to export"""

        if score_type not in ('NMI', 'Abs Corr'):
            print("score_type should be either 'NMI' or 'Abs Corr'")
            return
        
        if label not in self.sanmin.label_columns:
            print("label should be in the designated labels column")
            return

        if penalty_function_type not in ('None', 'Mean', 'Max'):
            print("penalty_function_type should be either 'None' or 'Mean' or 'Max'")
            return

        # get the correct matrix
        if score_type == 'Abs Corr':
            score_matrix = self.sanmin.abs_corr_matrix
        elif score_type == 'NMI':
            score_matrix = self.sanmin.NMI_matrix
            
        # get the correct penalty function
        if penalty_function_type == 'None':
            funct = self._return_zero
        elif penalty_function_type == 'Mean':
            funct = np.mean
        elif penalty_function_type == 'Max':
            funct = max

        # object to output
        feature_combos_with_score = list()

        # starting with each individual feature
        for feature in self.sanmin.final_features[label]:
            if feature in self.sanmin.label_columns: # don't add if it is a label
                continue

            # initial current combo
            curr_combo = [feature]

            # initial current score
            curr_combo_score = score_matrix.loc[feature][label]

            # initial combo appended with its score
            feature_combos_with_score.append((copy.deepcopy(curr_combo), curr_combo_score))

            switch = True

            while switch is True:

                # temporary scores
                curr_added_value = 0
                curr_feature_to_add = None

                # for all try-able combinations
                for candidate_feature in self.sanmin.final_features[label]: 
                    if candidate_feature in curr_combo or candidate_feature in self.sanmin.label_columns: # don't try those already in, nor those that are labels
                        continue
                    
                    # get candidate's own corr with label
                    candidate_feature_score = score_matrix.loc[candidate_feature][label]

                    # get list of corr between candidate and features currently in combo
                    candidate_feature_relation_scores = list()
                    for curr_combo_feature in curr_combo:
                        candidate_feature_relation_scores.append(score_matrix.loc[candidate_feature][curr_combo_feature])
                    
                    # if candidate score - penalty > current best, then accept; else, don't accept
                    if candidate_feature_score - funct(candidate_feature_relation_scores) >= curr_added_value:
                        curr_added_value = candidate_feature_score - funct(candidate_feature_relation_scores)
                        curr_feature_to_add = candidate_feature

                # if managed to add something to combo, then continue loop and add to overall list, else break this loop
                if curr_feature_to_add is None:
                    switch = False

                else:
                    curr_combo.append(curr_feature_to_add)
                    curr_combo_score += curr_added_value
                    feature_combos_with_score.append((copy.deepcopy(curr_combo), curr_combo_score))


        # Remove duplicates and sort
        feature_combos_with_score = self._features_duplicate_removal(feature_combos_with_score)
        feature_combos_with_score.sort(key = lambda x: x[1])

        # Get pure feature combinations, and scores of feature combinations
        feature_combo = [feature_combo[0] for feature_combo in feature_combos_with_score]
        feature_combo_scores = [feature_combo[1] for feature_combo in feature_combos_with_score]

        print(f"{len(feature_combos_with_score)} feature combinations, with combo scores ranging from {round(feature_combo_scores[0], 4)} to {round(feature_combo_scores[-1], 4)}")
        

        # Export combinations and scores as a json
        if export_address:
            json_output = {'feature_combos_with_score': feature_combos_with_score, 
                            'feature_combo': feature_combo, 
                            'feature_combo_scores': feature_combo_scores}
                            
            with open(export_address, 'w') as f:
                json.dump(json_output, f, indent=2) 
            print("Export Completed")
    
        return feature_combos_with_score, feature_combo, feature_combo_scores


    def _return_zero(self, dummy):
        """ Helper function that returns 0 for penalty, no matter input """
        return 0
    


    def _features_duplicate_removal(self, feature_combos_with_score):
        """ Helper function that remove duplicate combinations """
        for i in range(len(feature_combos_with_score)):
            feature_combos_with_score[i][0].sort()

        feature_combos_with_score.sort(key = lambda x:x[0])

        duplicate_i = []

        feature_combos_no_duplicates = []

        for i in range(len(feature_combos_with_score)-1):
            if i in duplicate_i:
                continue

            if feature_combos_with_score[i][0] == feature_combos_with_score[i+1][0]:
                # retain the higher score
                if feature_combos_with_score[i][1] >= feature_combos_with_score[i+1][1]:
                    duplicate_i.append(i)
                else:
                    duplicate_i.append(i+1)
            
            if i not in duplicate_i:
                feature_combos_no_duplicates.append(feature_combos_with_score[i])

        return feature_combos_no_duplicates