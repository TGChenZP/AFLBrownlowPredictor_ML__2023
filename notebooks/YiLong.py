# 20/02/2023





import pandas as pd
import copy





def combine_tuning_results(tuning_results, output_address):
    """ Combines multiple tuning result DataFrames into one DataFrame"""
    
    assert type(tuning_results) == list
    assert len(tuning_results) > 1

    columns = set(tuning_results[0].columns)
    for tuning_result in tuning_results[1:]:
        assert set(tuning_result.columns) == columns
        

    combined_tuning_results = pd.DataFrame()

    for tuning_result in tuning_results:
        combined_tuning_results = combined_tuning_results.append(tuning_result)
    
    output_address_split = output_address.split('.csv')[0]

    tuning_result.to_csv(f'{output_address_split}.csv', index = False)





class YiLong:



    def __init__(self, type):
        
        # check correct input
        assert type == 'Classification' or type == 'Regression' or 'GLM Regression'

        self.clf_type = type
        self._initialise_objects() # Initialise objects
        print(f'YiLong Initialised to analyse {self.clf_type}')



    def _initialise_objects(self):
        """ Helper to initialise objects """

        self.tuning_result = None
        self.hyperparameters = None
        self._seed = 18861201

        self.regression_extra_output_columns = ['Train r2', 'Val r2', 'Test r2', 
            'Train RMSE', 'Val RMSE', 'Test RMSE', 'Train MAPE', 'Val MAPE', 'Test MAPE', 'Time']
        self.classification_extra_output_columns = ['Train accu', 'Val accu', 'Test accu', 
            'Train balanced_accu', 'Val balanced_accu', 'Test balanced_accu', 'Train f1', 'Val f1', 'Test f1', 
            'Train precision', 'Val precision', 'Test precision', 'Train recall', 'Val recall', 'Test recall', 'Time']
        self.GLM_Regression_extra_output_columns = ['Train deviance', 'Val deviance', 'Test deviance', 'Time']

        self.discard_columns = None

        self.tuning_result = None
        self.hyperparameters = None

    

    def read_tuning_result(self, address, extra_to_discard_columns = None):
        """ Read in Tuning Result """

        if extra_to_discard_columns is not None:
            assert type(extra_to_discard_columns) == list

        self.tuning_result = pd.read_csv(address)

        print(f'Successfully read in tuning result, with {len(self.tuning_result)} columns')

        # get list of hyperparameters by taking what is not in the extra_output_columns
        if self.clf_type == 'Classification':
            self.discard_columns = copy.deepcopy(self.classification_extra_output_columns)
            
            if extra_to_discard_columns is not None:
                self.discard_columns.append(extra_to_discard_columns)


        elif self.clf_type == 'Regression':
            self.discard_columns = copy.deepcopy(self.regression_extra_output_columns)
            
            if extra_to_discard_columns is not None:
                self.discard_columns.append(extra_to_discard_columns)


        elif self.clf_type == 'GLM Regression':
            self.discard_columns = copy.deepcopy(self.GLM_Regression_extra_output_columns)
            
            if extra_to_discard_columns is not None:
                self.discard_columns.append(extra_to_discard_columns)

        self.hyperparameters = [col for col in self.tuning_result.columns if col not in self.discard_columns]



    def read_sorted_full_df(self, interested_statistic = None, ascending = False):
        """ View dataframe sorted in reverse in terms of validation score """
        
        assert type(ascending) == bool

        if self.tuning_result is None:
            print('Please run read_tuning_result() first')
            return
        
        if interested_statistic is not None:
            if self.clf_type == 'Regression':
                if interested_statistic not in self.regression_extra_output_columns:
                    print('Statistic not valid for a Regression Model')
                    return

            elif self.clf_type == 'Classification':
                if interested_statistic not in self.classification_extra_output_columns:
                    print('Statistic not valid for a Classification Model')
                    return
            
            elif self.clf_type == 'GLM Regression':
                if interested_statistic not in self.GLM_Regression_extra_output_columns:
                    print('Statistic not valid for a GLM Regression Model')
                    return
        
        

        if len(self.tuning_result) < 60:
            length = len(self.tuning_result)
        else:
            length = 60


        if self.clf_type =='Regression':
            if interested_statistic == None:
                interested_statistic = 'Val r2'

            sorted_tuning_results = self.tuning_result.sort_values([interested_statistic], ascending = ascending)

        elif self.clf_type =='Classification':
            if interested_statistic == None:
                interested_statistic = 'Val accu'
            
            sorted_tuning_results = self.tuning_result.sort_values([interested_statistic], ascending = ascending)
        
        elif self.clf_type =='GLM Regression':
            if interested_statistic == None:
                interested_statistic = 'Val deviance'
            
            sorted_tuning_results = self.tuning_result.sort_values([interested_statistic], ascending = ascending)

        

        sorted_tuning_results.index = range(len(sorted_tuning_results))
        best_hyperparameter_combination = {hyperparameter:sorted_tuning_results.iloc[0][hyperparameter] for hyperparameter in self.hyperparameters}

        # change layout of output
        out_best_hyperparameter_combination = [copy.deepcopy(best_hyperparameter_combination)]
        if 'features' in out_best_hyperparameter_combination[0]:
            out_best_hyperparameter_combination.extend([out_best_hyperparameter_combination[0]['features'], {'feature combo ningxiang score':out_best_hyperparameter_combination[0]['feature combo ningxiang score']}])
            del out_best_hyperparameter_combination[0]['features']
            del out_best_hyperparameter_combination[0]['feature combo ningxiang score']

        print('Best hyperameter combination:', out_best_hyperparameter_combination, '\n')

        print(f'Highest {length}')
        display(sorted_tuning_results.head(length))
        print(f'Lowest {length}')
        display(sorted_tuning_results.tail(length))


        return out_best_hyperparameter_combination 



    def read_mean_val_scores(self):
        """ View the means of evaluation metrics for combinations containing each individual value of a hyperparameter – for each hyperparameter """

        for col in self.hyperparameters: # for each hyperparameter
            
            if col == 'features': # report NingXiang score only
                    continue

            print('\nHYPERPARAMETER:', col.upper())

            hyperparameter_values = list(set(self.tuning_result[col]))
            hyperparameter_values.sort()
            
            # create this temporary dataframe
            validation_score_df = pd.DataFrame()
            for value in hyperparameter_values: # for each value in the hyperparameter

                tmp_df = self.tuning_result[self.tuning_result[col] == value] # select df with only those parameter values

                # get means
                if self.clf_type == 'Classification':
                    tmp_df_mean = tmp_df[self.classification_extra_output_columns[:-1]].mean().T
                elif self.clf_type == 'Regression':
                    tmp_df_mean = tmp_df[self.regression_extra_output_columns[:-1]].mean().T
                elif self.clf_type == 'GLM Regression':
                    tmp_df_mean = tmp_df[self.GLM_Regression_extra_output_columns[:-1]].mean().T

                # get number of observations in this group
                tmp_df_mean['n'] = len(tmp_df)

                # append to this temporary dataframe
                validation_score_df[f'{value}'] = tmp_df_mean

            display(validation_score_df)
    


    def read_grouped_scores(self):
        """ View all evaluation metrics for combinations grouped by containing each individual value of a hyperparameter – for each hyperparameter 
        If any of the individual values of a hyperparameter exceeds 60, then sample down to 60 without replacement """

        for col in self.hyperparameters: # for each hyperparameter

            if col == 'features': # report NingXiang score only
                    continue

            print('\nHYPERPARAMETER:', col.upper())

            hyperparameter_values = list(set(self.tuning_result[col]))
            hyperparameter_values.sort()
            
            # create this temporary dataframe
            for value in hyperparameter_values: # for each value in the hyperparameter

                print(f'{col} Value:', value)
                tmp_df = self.tuning_result[self.tuning_result[col] == value] # select df with only those parameter values

                if len(tmp_df) < 60:
                    length = len(tmp_df)
                else:
                    length = 60

                display(tmp_df.sample(length, replace=False, random_state=self._seed))
        """ View all evaluation metrics for combinations grouped by containing each individual value of a hyperparameter – for each hyperparameter 
        If any of the individual values of a hyperparameter exceeds 60, then sample down to 60 without replacement """

        for col in self.hyperparameters: # for each hyperparameter

            if col == 'features': # report NingXiang score only
                    continue

            print('\nHYPERPARAMETER:', col.upper())

            hyperparameter_values = list(set(self.tuning_result[col]))
            hyperparameter_values.sort()
            
            # create this temporary dataframe
            for value in hyperparameter_values: # for each value in the hyperparameter

                print(f'{col} Value:', value)
                tmp_df = self.tuning_result[self.tuning_result[col] == value] # select df with only those parameter values

                if len(tmp_df) < 60:
                    length = len(tmp_df)
                else:
                    length = 60

                display(tmp_df.sample(length, replace=False, random_state=self._seed))