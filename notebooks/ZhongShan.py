# 10/01/2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import pickle

from sklearn.metrics import normalized_mutual_info_score as NMI
from itertools import combinations 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def setup_project_directory(directories_to_create = ['notebooks', 
        'scripts', 
        'plots', 
        'models', 
        'data', 
        'data/raw', 
        'data/curated', 
        'presentables',
        'can_delete']
    ):
        """ Function to setup directory for a new project """

        for directory in directories_to_create:
            if not os.path.exists(f'./{directory}'):
                os.makedirs(f'./{directory}')



def create_directories(directories_to_create):
        """ Function to setup new directory for a new project. 
        Must write full relative directory! """

        if type(directories_to_create) is list:
            for directory in directories_to_create:
                if not os.path.exists(directory):
                    os.makedirs(directory)
        
        elif type(directories_to_create) is str:
            if not os.path.exists(directories_to_create):
                    os.makedirs(directories_to_create)



class ZhongShan:



    def __init__(self, full_data, toggle_index = True):
        """ Initiation: read in DataFrame """

        if type(full_data) != pd.core.frame.DataFrame:
            raise TypeError(f"Must input DataFrame; you inputted {type(full_data)}")
        
        full_data_copy = copy.deepcopy(full_data)

        self.full_data = full_data_copy
        print('Pandas DataFrame readin successful')

        if toggle_index:
            self.full_data = self._reset_index(self.full_data)
            print('Reset index successful')
        else:
            print('Did not reset index upon request')

        self._initialise_objects()



    def _initialise_objects(self):
        """ Helper to initialise all ZhongShan objects """

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.OHE_storage = None
        self.label_columns = None 
        self.index_columns = None
        self.discarded_columns = None
        self.feature_columns = None
        self.retained_columns = None
        self.numeric_cols = None
        self.non_numeric_cols = None
        self.full_data_IDE = None
        self.full_data_IDE_T = None
        self.pca = None
        self.pca_explained_variance_ratio = None
        self.final_ncomponents = None
        self.standardiser_objects = None
        self.corr_matrix = None
        self.abs_corr_matrix = None
        self.corr_heatmap = None
        self.NMI_matrix = None
        self.nmi_heatmap = None
        self.cont_scatter_plot = None
        self.cat_scatter_plot = None
        self.final_features = None
        self.feature_selected_full_data = None
        self.feature_selected_train_data = None
        self.feature_selected_val_data = None
        self.feature_selected_test_data = None


    
    def _reset_index(self, df):
        """ Helper function to reset index """

        df.index = range(len(df))
        return df
    


    def fill_na(self, df_name, fill_value=0):
        """ Fills na values in data with default value 0 """

        if df_name == 'Full':
            if self.full_data is None:
                print("Please input Full Data before using this function")
                return
            else:
                self.full_data = self.full_data.fillna(fill_value)
        
        elif df_name == 'Train':
            if self.train_data is None:
                print("Please input Train Data before using this function")
                return
            else:
                self.train_data = self.train_data.fillna(fill_value)

        elif df_name == 'Validate':
            if self.val_data is None:
                print("Please input Validate Data before using this function")
                return
            else:
                self.val_data = self.val_data.fillna(fill_value)

        elif df_name == 'Test':
            if self.test_data is None:
                print("Please input Test Data before using this function")
                return
            else:
                self.test_data = self.test_data.fillna(fill_value)
        
        print(f'Filled null values on {df_name} dataset with {fill_value}')
    


    def one_hot_encode_fit_transform(self, col_to_ohe, output_col_name):
        """ Fit and OHE transform one column of data based on full data """

        self._one_hot_encode_fit(col_to_ohe, output_col_name)
        self.one_hot_encode_transform('Full', col_to_ohe)

        print(f"Successfully fitted and OHE transformed Full data's '{col_to_ohe}' column")



    def _one_hot_encode_fit(self, col_to_ohe, output_col_name):
        """ Helper function to fit one column using OHE using full data """

        OHE = OneHotEncoder()
        OHE.fit(self.full_data[[col_to_ohe]])
        try:
            self.OHE_storage[col_to_ohe] = {'OHE_object': OHE, 'output_col_names': output_col_name}
        except:
            self.OHE_storage = {col_to_ohe: {'OHE_object': OHE, 'output_col_names': output_col_name}}
        print(f"Successfully fitted OHE on '{col_to_ohe}' column")



    def one_hot_encode_transform(self, df_name, col_to_ohe): 
        """ OHE transform one column of data using pre-trained OHE object """

        if self.OHE_storage is None:
            print("Please fit OHE using .one_hot_encode_fit_transform() before re-attempting")
            return
        
        if col_to_ohe not in self.OHE_storage:
            print(f"Please fit OHE to column '{col_to_ohe}' using .one_hot_encode_fit_transform() before re-attempting  transform")

        if df_name == 'Full':
            if self.full_data is None:
                print("Please input Full Data before using this function")
                return
            else:
                OHE_output = pd.DataFrame.sparse.from_spmatrix(\
                    self.OHE_storage[col_to_ohe]['OHE_object'].transform(self.full_data[[col_to_ohe]]))

                for i in range(len(self.OHE_storage[col_to_ohe]['output_col_names'])):
                    self.full_data[self.OHE_storage[col_to_ohe]['output_col_names'][i]] = list(OHE_output[i])
                
                self.full_data = self.full_data.drop([col_to_ohe], axis = 1)

        elif df_name == 'Train':
            if self.train_data is None:
                print("Please input Train Data before using this function")
                return
            else:
                OHE_output = pd.DataFrame.sparse.from_spmatrix(\
                    self.OHE_storage[col_to_ohe]['OHE_object'].transform(self.train_data[[col_to_ohe]]))

                for i in range(len(self.OHE_storage[col_to_ohe]['output_col_names'])):
                    self.train_data[self.OHE_storage[col_to_ohe]['output_col_names'][i]] = list(OHE_output[i])
                
                self.train_data = self.train_data.drop([col_to_ohe], axis = 1)

        elif df_name == 'Validate':
            if self.val_data is None:
                print("Please input Validate Data before using this function")
                return
            else:
                OHE_output = pd.DataFrame.sparse.from_spmatrix(\
                    self.OHE_storage[col_to_ohe]['OHE_object'].transform(self.val_data[[col_to_ohe]]))

                for i in range(len(self.OHE_storage[col_to_ohe]['output_col_names'])):
                    self.val_data[self.OHE_storage[col_to_ohe]['output_col_names'][i]] = list(OHE_output[i])

                self.val_data = self.val_data.drop([col_to_ohe], axis = 1)

        elif df_name == 'Test':
            if self.test_data is None:
                print("Please input Test Data before using this function")
                return
            else:
                OHE_output = pd.DataFrame.sparse.from_spmatrix(\
                    self.OHE_storage[col_to_ohe]['OHE_object'].transform(self.test_data[[col_to_ohe]]))

                for i in range(len(self.OHE_storage[col_to_ohe]['output_col_names'])):
                    self.test_data[self.OHE_storage[col_to_ohe]['output_col_names'][i]] = list(OHE_output[i])
                
                self.test_data = self.test_data.drop([col_to_ohe], axis = 1)
        
        print(f"OHE'ed and Dropped '{col_to_ohe}' column on {df_name} Data")
    


    def set_columns(self, label_columns, index_columns, discarded_columns): 
        """ Manually set which columns are for what purpose """
        
        self.label_columns = label_columns
        self.index_columns = index_columns
        self.discarded_columns = discarded_columns
        
        print(f'Successfully set label columns, consisting {len(self.label_columns)} columns')
        print(f'Successfully set index columns, consisting {len(self.index_columns)} columns')
        print(f'Successfully set discarded columns, consisting {len(self.discarded_columns)} columns')


        self.feature_columns = self._get_feature_columns()
        self.retained_columns = self._get_retained_columns()

        print(f'Successfully set feature columns, consisting {len(self.feature_columns)} columns')
        print(f'Successfully set retained columns, consisting {len(self.retained_columns)} columns')



    def _get_feature_columns(self): 
        """ Helper function to generate feature columns based on other manually set columns """

        feature_columns = [col for col in self.full_data.columns if col not in self.label_columns and \
            col not in self.index_columns and col not in self.discarded_columns]
        return feature_columns



    def _get_retained_columns(self): 
        """ Helper function to generate retained columns based on other manually set columns """

        retained_columns = copy.deepcopy(self.feature_columns)
        retained_columns.extend(self.label_columns)

        return retained_columns



    def view_setted_columns(self): 
        """ Print out the column purposes that are currently set """

        print("label columns:", self.label_columns, '\n')
        print("index columns:", self.index_columns, '\n')
        print("discarded columns:", self.discarded_columns, '\n')
        print("feature columns:", self.feature_columns, '\n')
        print("retained columns:", self.retained_columns, '\n')



    def basic_overview(self, df_name, view_how_many=10): 
        """ Head() and Tail() the current data (we choose which it is), and print out number of rows and columns """

        if df_name == 'Full':
            if self.full_data is None:
                print("Please input Full Data before using this function")
                return
            else:
                display(self.full_data.head(view_how_many))
                display(self.full_data.tail(view_how_many))

                print(f'Number of Rows (Instances*): {self.full_data.shape[0]}')
                print(f'Number of Columns (Features*): {self.full_data.shape[1]}')
        
        elif df_name == 'Train':
            if self.train_data is None:
                print("Please input Train Data before using this function")
                return
            else:
                display(self.train_data.head(view_how_many))
                display(self.train_data.tail(view_how_many))

                print(f'Number of Rows (Instances*): {self.train_data.shape[0]}')
                print(f'Number of Columns (Features*): {self.train_data.shape[1]}')

        elif df_name == 'Validate':
            if self.val_data is None:
                print("Please input Validate Data before using this function")
                return
            else:
                display(self.val_data.head(view_how_many))
                display(self.val_data.tail(view_how_many))

                print(f'Number of Rows (Instances*): {self.val_data.shape[0]}')
                print(f'Number of Columns (Features*): {self.val_data.shape[1]}')

        elif df_name == 'Test':
            if self.test_data is None:
                print("Please input Test Data before using this function")
                return
            else:
                display(self.test_data.head(view_how_many))
                display(self.test_data.tail(view_how_many))

                print(f'Number of Rows (Instances*): {self.test_data.shape[0]}')
                print(f'Number of Columns (Features*): {self.test_data.shape[1]}')



    def get_full_data_analysis(self):
        """ Compute basic summaries on Full Data """

        if self.retained_columns is None:
            print('Please run .set_columns() before re-attempting this method')
            return

        count_null = list()
        for col in self.retained_columns:
            count_null.append(self.full_data[col].isna().sum())

        self.full_data_IDE = pd.DataFrame({'Number of Missing': count_null}, index = self.retained_columns)

        self.numeric_cols = list()
        self.non_numeric_cols = list()

        mean = list()
        std = list()
        min = list()
        q1 = list()
        q2 = list()
        q3 = list()
        max = list()

        for col in self.retained_columns:
            if self.full_data[col].dtype in [np.float64, np.int64]:
                self.numeric_cols.append(col)
                describe = self.full_data[col].describe()
                
                mean.append(describe['mean'])
                std.append(describe['std'])
                min.append(describe['min'])
                q1.append(describe['25%'])
                q2.append(describe['50%'])
                q3.append(describe['75%'])
                max.append(describe['max'])
            
            else:
                self.non_numeric_cols.append(col)
                mean.append(np.nan)
                std.append(np.nan)
                min.append(np.nan)
                q1.append(np.nan)
                q2.append(np.nan)
                q3.append(np.nan)
                max.append(np.nan)

        self.full_data_IDE['mean'] = mean
        self.full_data_IDE['std'] = std
        self.full_data_IDE['min'] = min
        self.full_data_IDE['q1'] = q1
        self.full_data_IDE['q2'] = q2
        self.full_data_IDE['q3'] = q3
        self.full_data_IDE['max'] = max

        self.full_data_IDE['IQR'] = self.full_data_IDE['q3'] - self.full_data_IDE['q1']
        self.full_data_IDE['1.5 upper bound'] = self.full_data_IDE['q3'] + 1.5*self.full_data_IDE['IQR']
        self.full_data_IDE['1.5 lower bound'] = self.full_data_IDE['q1'] - 1.5*self.full_data_IDE['IQR']
        self.full_data_IDE['3 upper bound'] = self.full_data_IDE['q3'] + 3*self.full_data_IDE['IQR']
        self.full_data_IDE['3 lower bound'] = self.full_data_IDE['q1'] - 3*self.full_data_IDE['IQR']

        outlier1 = list()
        outlier3 = list()

        for col in self.retained_columns:
            if self.full_data[col].dtype in [np.float64, np.int64]:
                outlier1.append(len(self.full_data[(self.full_data[col] > self.full_data_IDE.loc[col]['1.5 upper bound']) | (self.full_data[col] < self.full_data_IDE.loc[col]['1.5 lower bound'])]))
                outlier3.append(len(self.full_data[(self.full_data[col] > self.full_data_IDE.loc[col]['3 upper bound']) | (self.full_data[col] < self.full_data_IDE.loc[col]['3 lower bound'])]))

            else:
                outlier1.append(np.nan)
                outlier3.append(np.nan)

        self.full_data_IDE['number of 1.5 outliers'] = outlier1
        self.full_data_IDE['number of 3 outliers']  = outlier3

        self.full_data_IDE_T = self.full_data_IDE.T

        print("Got Full Data Analysis")
        


    def view_column_types(self):
        """ Print out the column types of the current Full Data """

        if self.numeric_cols is None or self.non_numeric_cols is None:
            print('Please run .get_full_data_analysis() before re-attempting this method')
            return

        print("numeric columns:", self.numeric_cols, '\n')
        print("non-numeric columns:", self.non_numeric_cols, '\n')

    

    def get_boxplot(self, col):
        """ Generate boxplots of a column using full data """

        if self.full_data_IDE_T is None:
            print('Please run .get_full_data_analysis() before re-attempting this method')
            return

        fig, axes = plt.subplots(nrows=1, ncols=1) 
        plt.boxplot(self.full_data[col]);
        plt.title(f'{col} Boxplot')
        plt.xticks([1], [col])
        plt.yticks(list(self.full_data_IDE_T[col][['min', '1.5 lower bound', 'q1', 'q2', 'q3', '1.5 upper bound', 'max']]))
        plt.show()

        try:
            self.boxplot_objects[col] = fig
        except:
            self.boxplot_objects = {col: fig}



    def view_full_data_analysis(self):
        """ Print the full Full Data IDE dataframe """

        if self.full_data_IDE is None:
            print('Please run .get_full_data_analysis() before re-attempting this method')
            return

        display(self.full_data_IDE)



    def view_full_data_col_analysis(self, col):
        """ For one retained column, print out the Full Data summary and also boxplot """

        if self.retained_columns is None:
            print('Please run .set_columns() before re-attempting this method')
            return

        if self.full_data_IDE_T is None:
            print('Please run .get_full_data_analysis() before re-attempting this method')
            return

        if col in self.retained_columns:
            print(f"Full Data IDE on column '{col}'")
            display(self.full_data_IDE_T[[col]])
            self.get_boxplot(col)
        else:
            print("Column not in retained columns")
    


    def view_all_full_data_col_analysis(self):
        """ Print out all Full Data summary column by column and also boxplot """

        if self.retained_columns is None:
            print('Please run .set_columns() before re-attempting this method')
            return

        for col in self.retained_columns:
            self.view_full_data_col_analysis(col)



    def read_in_train_test_split(self, train_data, val_data, test_data):
        """ Read in Train Test Split data """

        self.train_data = train_data
        print('Train Data read in successfully')
        self.val_data = val_data
        print('Validation Data read in successfully')
        self.test_data = test_data
        print('Test Data read in successfully')
    


    def train_test_split(self, train_percentage, val_percentage, test_percentage, random = True, seed = 18661112):
        """ Perform Train Test Split on Full Data that is already in ZhongShan """

        assert train_percentage + val_percentage + test_percentage == 1
        assert train_percentage <= 1 and 0 <= train_percentage
        assert val_percentage <= 1 and 0 <= val_percentage
        assert test_percentage <= 1 and 0 <= test_percentage

        self.train_data, val_test_data = train_test_split(self.full_data, train_size = train_percentage, test_size = 1-train_percentage, shuffle = random, random_state = seed)
        self.val_data, self.test_data = train_test_split(val_test_data, train_size = val_percentage, test_size =test_percentage,  shuffle = random, random_state= seed)
    


    def pca_fit(self, n_components = 10):
        """ fit PCA using training data """
        
        self.pca_n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.train_data[self.feature_columns])

        self.pca_explained_variance_ratio = self.pca.explained_variance_ratio_

        print("PCA successfully fitted on training data")



    def view_pca_explained_variance_ratio(self):
        """ View dataframe that stores PCA explained variance """

        if self.pca_explained_variance_ratio is None:
            print("Please fit PCA object using .pca_fit() before re-attempting")
            return

        print("PCA Explained Variance")
        display(pd.DataFrame({'explained_variance_ratio':list(self.pca_explained_variance_ratio)}, 
            index = range(len(self.pca_explained_variance_ratio))))
        


    def pca_set_final_ncomponents(self, final_ncomponents):
        """ Set how many dimensions of PCA we want to use in the end """

        if self.pca is None:
            print("Please fit PCA using .pca_fit() and view explained variance using .view_pca_explained_variance_ratio() before setting final_ncomponents")
            return

        self.final_ncomponents = final_ncomponents
        print(f'Using first {final_ncomponents} PCA components')



    def pca_transform(self, df_name): 
        """ PCA transform data using pre-trained PCA object """

        if self.pca is None:
            print("Please fit PCA object using .pca_fit() before re-attempting")
            return


        if df_name == 'Full':
            if self.full_data is None:
                print("Please input Full Data before using this function")
                return
            else:
                pca_tmp = self.pca.transform(self.full_data[self.feature_columns])
            
                pca_output = pd.DataFrame(pca_tmp)
                pca_output.columns = [f'PCA {i}' for i in range(self.pca_n_components)]

                for i in range(self.final_ncomponents):
                    self.full_data[f'PCA {i}'] = pca_output[f'PCA {i}']

        elif df_name == 'Train':
            if self.train_data is None:
                print("Please input Train Data before using this function")
                return
            else:
                pca_tmp = self.pca.transform(self.train_data[self.feature_columns])
            
                pca_output = pd.DataFrame(pca_tmp)
                pca_output.columns = [f'PCA {i}' for i in range(self.pca_n_components)]

                for i in range(self.final_ncomponents):
                    self.train_data[f'PCA {i}'] = pca_output[f'PCA {i}']

        elif df_name == 'Validate':
            if self.val_data is None:
                print("Please input Validate Data before using this function")
                return
            else:
                pca_tmp = self.pca.transform(self.val_data[self.feature_columns])
            
                pca_output = pd.DataFrame(pca_tmp)
                pca_output.columns = [f'PCA {i}' for i in range(self.pca_n_components)]

                for i in range(self.final_ncomponents):
                    self.val_data[f'PCA {i}'] = pca_output[f'PCA {i}']

        elif df_name == 'Test':
            if self.test_data is None:
                print("Please input Test Data before using this function")
                return
            else:
                pca_tmp = self.pca.transform(self.test_data[self.feature_columns])
            
                pca_output = pd.DataFrame(pca_tmp)
                pca_output.columns = [f'PCA {i}' for i in range(self.pca_n_components)]

                for i in range(self.final_ncomponents):
                    self.test_data[f'PCA {i}'] = pca_output[f'PCA {i}']
    
        print(f'PCA transformed {df_name} Data, PLEASE REMEMBER TO USE .pca_update_features() to update feature_columns')
        


    def pca_transform_all(self):
        """ One step PCA transform all data """

        for df_name in ['Full', 'Train', 'Validate', 'Test']:
            self.pca_transform(df_name)
    


    def pca_update_features(self):
        """ Update features list after adding PCA features """

        for i in range(self.final_ncomponents):
            self.feature_columns.append(f'PCA {i}')
        
        self._get_retained_columns()

        print(f'Updated feature columns and retained columns')


    
    def standardise_fit(self):
        """ Fit standardiser using Full Data """

        if self.train_data is None:
            print('Please input Train Data before re-attempting')
            return

        self.standardiser_objects = dict()

        for col in self.retained_columns:
            standardiser = StandardScaler()
            standardiser.fit(self.train_data[[col]])

            self.standardiser_objects[col] = standardiser



    def standardise_transform(self, df_name):
        """ Transform data using pre-fitted standardiser """

        if self.standardiser_objects is None:
            print("Please use .standardise_fit() to fit standardisers before re-attempting")
            return

        if df_name == 'Full':
            
            if self.full_data is None:
                print("Please input Full Data before using this function")
                return
            else:

                for col in self.retained_columns:
                    standardiser = self.standardiser_objects[col]
                    standardiser_output = standardiser.transform(self.full_data[[col]])
                    standardiser_list_output = [x[0] for x in standardiser_output]

                    self.full_data[col] = standardiser_list_output
        
        elif df_name == 'Train':
            
            if self.train_data is None:
                print("Please input Train Data before using this function")
                return
            else:
                for col in self.retained_columns:
                    standardiser = self.standardiser_objects[col]
                    standardiser_output = standardiser.transform(self.train_data[[col]])
                    standardiser_list_output = [x[0] for x in standardiser_output]

                    self.train_data[col] = standardiser_list_output
        
        elif df_name == 'Validate':

            if self.val_data is None:
                print("Please input Validate Data before using this function")
                return
            else:
                for col in self.retained_columns:
                    standardiser = self.standardiser_objects[col]
                    standardiser_output = standardiser.transform(self.val_data[[col]])
                    standardiser_list_output = [x[0] for x in standardiser_output]

                    self.val_data[col] = standardiser_list_output

        elif df_name == 'Test':
            
            if self.test_data is None:
                print("Please input Test Data before using this function")
                return
            else:
                for col in self.retained_columns:
                    standardiser = self.standardiser_objects[col]
                    standardiser_output = standardiser.transform(self.test_data[[col]])
                    standardiser_list_output = [x[0] for x in standardiser_output]

                    self.test_data[col] = standardiser_list_output

        print(f'Standardised all retained columns in {df_name} Data')



    def standardise_transform_all(self):
        """ Standardise all data """

        for df_name in ['Full', 'Train', 'Validate', 'Test']:
            self.standardise_transform(df_name)
    
    
    
    def get_abs_corr(self):
        """ Calculate correlation and absolute Correlation Matrix on Training Data """

        if self.train_data is None:
            print('Please input Train Data before using this function')
            return
        
        if self.retained_columns is None:
            print("Please run .set_columns() before re-attempting")
            return

        self.corr_matrix = self.train_data[self.retained_columns].corr(method='pearson')
        self.abs_corr_matrix = self.corr_matrix.apply(abs)

        print("Calculated correlation and absolute correlation matrix on Train data")
    


    def view_corr_matrix(self):
        """ View the correlation matrix """
        
        if self.corr_matrix is None:
            print('Please run .get_abs_corr() before re-attempting')
            return

        display(self.corr_matrix)



    def view_abs_corr_matrix(self):
        """ View the absolute correlation matrix """

        if self.abs_corr_matrix is None:
            print('Please run .get_abs_corr() before re-attempting')
            return

        display(self.abs_corr_matrix)

    

    def get_corr_heatmap(self):
        """ View the absolute correlation heatmap """

        if self.abs_corr_matrix is None:
            print('Please run .get_abs_corr() before re-attempting')
            return

        fig, axes = plt.subplots(figsize=(20,20), nrows=1, ncols=1, dpi=80) 
        sns.heatmap(self.abs_corr_matrix)
        plt.show()
        self.corr_heatmap = fig
    


    def view_top_corr(self):
        """ View the sorted absolute correlation for features on each label """

        if self.abs_corr_matrix is None:
            print('Please run .get_abs_corr() before re-attempting')
            return

        for col in self.label_columns:
            print(f'Correlation between features and {col}')
            display(self.abs_corr_matrix.sort_values(by=col, ascending=False)[[col]].reset_index().head(60))
    


    def get_nmi(self):
        """ Calculate the NMI Matrix on Training data """

        if self.train_data is None:
            print('Please input Train Data before using this function')
            return
        
        if self.retained_columns is None:
            print("Please run .set_columns() before re-attempting")
            return

        self.NMI_matrix = pd.DataFrame({col:[0.0 for i in range(len(self.retained_columns))] for col in self.retained_columns}, index = self.retained_columns)

        combos = combinations(self.retained_columns, 2)

        for combo in combos:
            nmi = NMI(self.train_data[combo[0]], self.train_data[combo[1]])
            self.NMI_matrix.loc[combo[0]][combo[1]] = nmi
            self.NMI_matrix.loc[combo[1]][combo[0]] = nmi

        for col in self.retained_columns:
            self.NMI_matrix.loc[col][col] = 1
    


    def view_nmi_matrix(self):
        """ View the NMI matrix """

        if self.NMI_matrix is None:
            print('Please run .get_nmi() before re-attempting')
            return

        display(self.NMI_matrix)
    


    def get_nmi_heatmap(self):
        """ View the NMI heatmap """

        if self.NMI_matrix is None:
            print('Please run .get_nmi() before re-attempting')
            return

        fig, axes = plt.subplots(figsize=(20,20), nrows=1, ncols=1, dpi=80) 
        sns.heatmap(self.NMI_matrix)
        self.nmi_heatmap = fig
            
    

    def view_top_nmi(self):
        """ View sorted NMI for features on each label """

        if self.NMI_matrix is None:
            print('Please run .get_nmi() before re-attempting')
            return

        for col in self.label_columns:
            print(f'NMI between features and {col}')
            display(self.NMI_matrix.sort_values(by=col, ascending=False)[[col]].reset_index().head(60))

    

    def continuous_scatter_plot(self, col):
        """ Plot scatter plot with feature on x axis and label on y axis """

        if self.train_data is None:
            print('Please run .get_nmi() before re-attempting')
            return

        if self.label_columns is None:
            print('Please run .get_nmi() before re-attempting')
            return

        for label in self.label_columns:
            fig, axes = plt.subplots(nrows=1, ncols=1, dpi=80) 
            plt.title(f'{col} v {label}')
            plt.ylabel(label)
            plt.xlabel(col)
            plt.scatter(self.train_data[col], self.train_data[label])
            plt.show()

            try:
                self.cont_scatter_plot[f'{col}:{label}'] = fig
            except:
                self.cont_scatter_plot = {f'{col}:{label}': fig}
            


    def categorical_scatter_plot(self, col):
        """ Plot scatter plot with label on x axis and feature on y axis """

        for label in self.label_columns:
            fig, axes = plt.subplots(nrows=1, ncols=1, dpi=80) 
            plt.title(f'{label} v {col}')
            plt.ylabel(col)
            plt.xlabel(label)
            plt.scatter(self.train_data[label], self.train_data[col])
            plt.show()

            try:
                self.cat_scatter_plot[f'{col}:{label}'] = fig
            except:
                self.cat_scatter_plot = {f'{col}:{label}': fig}
    
    

    def feature_selection(self, cutoffs):
        """ Read in feature selection cutoff value """

        self.feature_selection_cutoffs = cutoffs
        print(f"Read in feature selection cutoff value at {self.feature_selection_cutoffs}")



    def view_feature_label_analysis(self, col, scatter_type='continuous'):
        """ View feature-label analysis for one feature """
        
        if self.NMI_matrix is None:
            print('Please run .get_nmi() before re-attempting')
            return

        if col in self.retained_columns:
            print(f"Feature-Label Data IDE on column '{col}'")
            display(self.abs_corr_matrix.loc[col][self.label_columns])
            display(self.NMI_matrix.loc[col][self.label_columns])
            if scatter_type == 'continuous':
                self.continuous_scatter_plot(col)
            else:
                self.categorical_scatter_plot(col)
        else:
            print("Column not in retained columns")
    


    def view_all_feature_label_analysis(self, scatter_type = 'continuous'):
        """ Print the feature-label analysis for all features """

        for col in self.feature_columns:
            self.view_feature_label_analysis(col, scatter_type = scatter_type)

    
    
    def export_plot(self, scatter_type, feature, label, address):
        """ save plots """
        
        address_split = address.split('.png')[0]

        if scatter_type == 'continuous':
            if f'{feature}:{label}' not in self.cont_scatter_plot:
                print("Plot does not exist - create it using various functions before re-attempting")
                return

            self.cont_scatter_plot[f'{feature}:{label}.png'].savefig(f'{address_split}.png')
        else:
            if f'{feature}:{label}' not in self.cat_scatter_plot:
                print("Plot does not exist - create it using various functions before re-attempting")
                return

            self.cat_scatter_plot[f'{feature}:{label}.png'].savefig(f'{address_split}.png')

        print("Plot saved successfully")



    def get_selected_features(self, metric = 'corr'):
        """ Extract lists that contain selected features for each label """

        if self.label_columns is None:
            print("Please run .set_columns() before re-attempting")
            return

        tmp_final_features = dict()
        self.final_features = dict()

        if metric == 'corr':
            if self.abs_corr_matrix is None:
                print("Please run .get_abs_corr() before re-attempting")
                return
            for label in self.label_columns:
                tmp_final_features[label] = list(self.abs_corr_matrix[self.abs_corr_matrix[label]>self.feature_selection_cutoffs].index)
        elif metric == 'nmi':
            if self.NMI_matrix is None:
                print("Please run .get_nmi() before re-attempting")
                return

            for label in self.label_columns:
                tmp_final_features[label] = list(self.NMI_matrix[self.NMI_matrix[label]>self.feature_selection_cutoffs].index)

        for label in self.label_columns:
            tmp_lst = list()
            for col in tmp_final_features[label]:
                if col not in self.label_columns:
                    tmp_lst.append(col)
            self.final_features[label] = tmp_lst
        
        print("Successfully got selected features")
        


    def get_feature_selected_data(self):
        """ Apply the columns to our data """

        if self.full_data is None:
            print("Please input Full Data")
            return
        if self.train_data is None:
            print("Please input Train Data")
            return
        if self.val_data is None:
            print("Please input Validation Data")
            return
        if self.test_data is None:
            print("Please input Test Data")
            return

        if self.feature_columns is None:
            print("Please run .set_columns() before re-attempting")
            return

        self.feature_selected_full_data = dict()
        self.feature_selected_train_data = dict()
        self.feature_selected_val_data = dict()
        self.feature_selected_test_data = dict()

        for label in self.label_columns:
            feature_columns = self.final_features[label]
            feature_columns.append(label)
            
            self.feature_selected_full_data[label] = self.full_data[feature_columns]
            self.feature_selected_train_data[label] = self.train_data[feature_columns]
            self.feature_selected_val_data[label] = self.val_data[feature_columns]
            self.feature_selected_test_data[label] = self.test_data[feature_columns]



    def export_data(self, df_name, label, address, index = False):
        """ Export DataFrame """

        if self.feature_selected_full_data is None:
            print('Please run .get_feature_selected_data() before re-attempting')
            return

        address_split = address.split('.csv')[0]

        if df_name == "Full":
            self.feature_selected_full_data[label].to_csv(f'{address_split}.csv', index=index)
        
        elif df_name == "Train":
            self.feature_selected_train_data[label].to_csv(f'{address_split}.csv', index=index)

        elif df_name == "Validate":
            self.feature_selected_val_data[label].to_csv(f'{address_split}.csv', index=index)

        elif df_name == "Test":
            self.feature_selected_test_data[label].to_csv(f'{address_split}.csv', index=index)


    
    def export_SanMin_components(self, address):
        """ Export Components of Sanmin as a dictionary in a pickle """

        sanmin_components = {
            "OHE_storage": copy.deepcopy(self.OHE_storage),
            'pca': copy.deepcopy(self.pca),
            'final_ncomponents': copy.deepcopy(self.final_ncomponents),
            'standardiser_objects': copy.deepcopy(self.standardiser_objects),
            'final_features': copy.deepcopy(self.final_features),
            'retained_columns': copy.deepcopy(self.retained_columns),
            'label_columns': copy.deepcopy(self.label_columns),
            'abs_corr_matrix': copy.deepcopy(self.abs_corr_matrix),
            'NMI_matrix': copy.deepcopy(self.NMI_matrix),
        }

        address_split = address.split('.pickle')[0]

        with open(f'{address}.pickle', 'wb') as f:
            pickle.dump(sanmin_components, f)





class SanMin:

    def __init__(self, input, input_type):
        
        if input_type not in ('ZhongShan', 'Components'):
            print('input_type must be either "ZhongShan" or "Components"')
            return 

        if input_type == 'ZhongShan':
            self.OHE_storage = copy.deepcopy(input.OHE_storage)
            self.pca = copy.deepcopy(input.pca)
            self.final_ncomponents = copy.deepcopy(input.final_ncomponents)
            self.standardiser_objects = copy.deepcopy(input.standardiser_objects)
            self.final_features = copy.deepcopy(input.final_features)
            self.retained_columns = copy.deepcopy(input.retained_columns)
            self.label_columns = copy.deepcopy(input.label_columns)
            self.abs_corr_matrix = copy.deepcopy(input.abs_corr_matrix)
            self.NMI_matrix = copy.deepcopy(input.NMI_matrix)
            
        elif input_type == 'Components':

            with open(f'{input}', 'rb') as f:
                sanmin_components = pickle.load(f)
            
            self.OHE_storage = sanmin_components['OHE_storage']
            self.pca = sanmin_components['OHE_storage']
            self.final_ncomponents = sanmin_components['final_ncomponents']
            self.standardiser_objects = sanmin_components['standardiser_objects']
            self.final_features = sanmin_components['final_features']
            self.retained_columns = sanmin_components['retained_columns']
            self.label_columns = sanmin_components['label_columns']
            self.abs_corr_matrix = sanmin_components['abs_corr_matrix']
            self.NMI_matrix = sanmin_components['NMI_matrix']
        

        self.feature_selected_future_data = None



    def import_future_data(self, future_data, toggle_index = True):
        """ Read in Future Data for transformation """

        self.future_data = future_data

        if toggle_index:
            self.future_data = self.reset_index(self.future_data)
            print('Reset index successful')
        else:
            print('Did not reset index upon request')



    def export_data(self, label, address, index=False):
        """ export the manipulated future data """

        if self.feature_selected_future_data is None:
            print('Please run .get_feature_selected_data() before re-attempting')
            return
        
        address_split = address.split('.csv')[0]
        
        self.feature_selected_future_data[label].to_csv(f'{address_split}.csv', index=index)



    def standardise_transform(self):
        """ Transform data using pre-fitted standardiser """

        if self.standardiser_objects is None:
            print("No standardiser in this SanMin")
            return

        if self.future_data is None:
            print("Please input Future Data before using this function")
            return
        else:
            for col in self.retained_columns:
                standardiser = self.standardiser_objects[col]
                standardiser_output = standardiser.transform(self.future_data[[col]])
                standardiser_list_output = [x[0] for x in standardiser_output]

                self.future_data[col] = standardiser_list_output

        print(f'Standardised all retained columns in Future Data')
    


    def fill_na(self, fill_value=0): 
        """ Helper to fill na values in data with default value 0 """

        if self.future_data is None:
            print("Please input Future Data before using this function")
            return
        else:
            self.future_data = self.future_data.fillna(fill_value)
        
        print(f'Filled null values on Future Data dataset with {fill_value}')



    def one_hot_encode_transform(self, col_to_ohe): 
        """ OHE transform one column of data using pre-trained OHE object """

        if self.OHE_storage is None:
            print("No OHE in this SanMin")
            return
        
        if col_to_ohe not in self.OHE_storage:
            print(f"No OHE of this column in this SanMin")

        if self.future_data is None:
            print("Please input Full Data before using this function")
            return

        else:
            OHE_output = pd.DataFrame.sparse.from_spmatrix(\
                self.OHE_storage[col_to_ohe]['OHE_object'].transform(self.future_data[[col_to_ohe]]))

            for i in range(len(self.OHE_storage[col_to_ohe]['output_col_names'])):
                self.future_data[self.OHE_storage[col_to_ohe]['output_col_names'][i]] = list(OHE_output[i])
            
            self.future_data = self.future_data.drop([col_to_ohe], axis = 1)
        
        print(f"OHE'ed and Dropped '{col_to_ohe}' column on Future Data")



    def pca_transform(self): 
        """ PCA transform data using pre-trained PCA object """

        if self.pca is None:
            print("No PCA object in this SanMin")
            return

        if self.future_data is None:
            print("Please input Future Data before using this function")
            return
        else:
            pca_tmp = self.pca.transform(self.future_data[self.feature_columns])
        
            pca_output = pd.DataFrame(pca_tmp)
            pca_output.columns = [f'PCA {i}' for i in range(self.pca_n_components)]

            for i in range(self.final_ncomponents):
                self.future_data[f'PCA {i}'] = pca_output[f'PCA {i}']

        print(f'PCA transformed Future Data, PLEASE REMEMBER TO USE .pca_update_features() to update feature_columns')
            


    def get_feature_selected_data(self):
        """ Apply the pre-selected columns to Future Data data """

        if self.future_data is None:
            print("Please input Future Data")
            return

        if self.feature_columns is None:
            print("Please run .set_columns() before re-attempting")
            return

        self.feature_selected_future_data = dict()

        for label in self.feature_columns:
            feature_columns = self.final_features[label]
            feature_columns.append(label)
            
            self.feature_selected_future_data[label] = self.future_data[feature_columns]

    

    def view_abs_corr_matrix(self):
        """ View the absolute correlation matrix """

        if self.abs_corr_matrix is None:
            print('The ZhongShan object which produced this SanMin did not have run .get_abs_corr()')
            return

        display(self.abs_corr_matrix)



    def view_nmi_matrix(self):
        """ View the NMI matrix """

        if self.NMI_matrix is None:
            print('The ZhongShan object which produced this SanMin did not have run .get_nmi()')
            print('Please run .get_nmi() before re-attempting')
            return

        display(self.NMI_matrix)



    def export_SanMin(self, address):
        """ Exports SanMin object """
    
        address_split = address.split('.pickle')[0]

        with open(f'{address_split}.pickle', 'wb') as f:
            pickle.dump(self, f)