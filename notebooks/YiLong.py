import pandas as pd





class YiLong:



    def __init__(self, type):
        
        # check correct input
        assert type == 'Classification' or type == 'Regression'

        self.clf_type = type
        self._initialise_objects() # Initialise objects
        print(f'YiLong Initialised to analyse {self.clf_type}')



    def _initialise_objects(self):
        """ Helper to initialise objects """

        self.tuning_result = None
        self.hyperparameters = None

        self.regression_extra_output_columns = ['Train r2', 'Val r2', 'Test r2', 
            'Train RMSE', 'Val RMSE', 'Test RMSE', 'Train MAPE', 'Val MAPE', 'Test MAPE', 'Time']
        self.classification_extra_output_columns = ['Train accu', 'Val accu', 'Test accu', 
            'Train balanced_accu', 'Val balanced_accu', 'Test balanced_accu', 'Train f1', 'Val f1', 'Test f1', 
            'Train precision', 'Val precision', 'Test precision', 'Train recall', 'Val recall', 'Test recall', 'Time']
        self.tuning_result = None
        self.hyperparameters = None

    

    def read_tuning_result(self, address):
        """ Read in Tuning Result """

        self.tuning_result = pd.read_csv(address)

        print(f'Successfully read in tuning result, with {len(self.tuning_result)} columns')

        # get list of hyperparameters by taking what is not in the extra_output_columns
        if self.clf_type == 'Classification':
            self.hyperparameters = [col for col in self.tuning_result.columns if col not in self.classification_extra_output_columns]

        elif self.clf_type == 'Regression':
            self.hyperparameters = [col for col in self.tuning_result.columns if col not in self.regression_extra_output_columns]



    def read_sorted_full_df(self):
        """ View dataframe sorted in reverse in terms of validation score """
        
        if self.tuning_result is None:
            print('Please run read_tuning_result() first')

        if len(self.tuning_result) < 60:
            length = len(self.tuning_result)
        else:
            length = 60

        if self.clf_type =='Regression':
            print(f'Highest {length}')
            display(self.tuning_result.sort_values(['Val r2'], ascending = False).head(length))
            print(f'Lowest {length}')
            display(self.tuning_result.sort_values(['Val r2'], ascending = False).tail(length))
        elif self.clf_type =='Classification':
            print(f'Highest {length}')
            display(self.tuning_result.sort_values(['Val accu'], ascending = False).head(length))
            print(f'Lowest {length}')
            display(self.tuning_result.sort_values(['Val accu'], ascending = False).tail(length))

    

    def read_mean_val_scores(self):
        """ View the means of evaluation metrics for combinations containing each individual value of a hyperparameter – for each hyperparameter """

        for col in self.hyperparameters: # for each hyperparameter

            print(col)

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

                # get number of observations in this group
                tmp_df_mean['n'] = len(tmp_df)

                # append to this temporary dataframe
                validation_score_df[f'{value}'] = tmp_df_mean

            display(validation_score_df)
    


    def read_grouped_scores(self):
        """ View all evaluation metrics for combinations grouped by containing each individual value of a hyperparameter – for each hyperparameter 
        If any of the individual values of a hyperparameter exceeds 60, then sample down to 60 without replacement """

        for col in self.hyperparameters: # for each hyperparameter

            print(col)

            hyperparameter_values = list(set(self.tuning_result[col]))
            hyperparameter_values.sort()
            
            # create this temporary dataframe
            for value in hyperparameter_values: # for each value in the hyperparameter
                tmp_df = self.tuning_result[self.tuning_result[col] == value] # select df with only those parameter values

                if len(tmp_df) < 60:
                    length = len(tmp_df)
                else:
                    length = 60

                display(tmp_df.sample(length, replace=False))