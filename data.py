""" Functions for importing data for experiments. """

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
    
    
def preprocess_data_credit(output_dir, protected_column='EDUCATION', proxy_noises=[0.1,0.2,0.3,0.4,0.5]):
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/credit_default.csv")
    columns_to_read = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default']
    df = pd.read_csv(data_path, usecols=columns_to_read)
    if protected_column == 'AGE':
        df['AGE_BUCKETS'] = pd.qcut(df['AGE'], 4, labels=[1,2,3,4])
        df = pd.get_dummies(df, columns=['AGE_BUCKETS'])
        protected_columns = ['AGE_BUCKETS_1', 'AGE_BUCKETS_2', 'AGE_BUCKETS_3', 'AGE_BUCKETS_4']
    elif protected_column == 'EDUCATION':
        df = pd.get_dummies(df, columns=['EDUCATION'])
        df['EDUCATION_grad'] = df['EDUCATION_1']
        df['EDUCATION_uni'] = df['EDUCATION_2']
        df['EDUCATION_hs_other'] = df['EDUCATION_3'] + df['EDUCATION_4'] + df['EDUCATION_5'] + df['EDUCATION_6'] + df['EDUCATION_0']
        columns_to_keep = ['LIMIT_BAL', 'SEX', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default', 'EDUCATION_grad', 'EDUCATION_uni', 
           'EDUCATION_hs_other']
        df = df[columns_to_keep]
        protected_columns = ['EDUCATION_grad', 'EDUCATION_uni', 'EDUCATION_hs_other']
    
    # Generate proxy groups.
    for noise in proxy_noises:
        df = generate_proxy_columns(df, protected_columns, noise_param=noise)
    df.to_csv(path_or_buf=os.path.join(output_dir, "credit_default_processed.csv"), index=False)

    
def load_dataset_credit():
    """ Loads UCI credit default data from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/credit_default_processed.csv")
    df = pd.read_csv(data_path)
    # Quantize continuous features
    continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
       'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'] 
    for feature in continuous_features:
        df[feature] = pd.qcut(df[feature], 4, labels=range(4))
    categorical_features = ['MARRIAGE', 'SEX', 'PAY_0', 'PAY_2', 'PAY_3',
       'PAY_4', 'PAY_5', 'PAY_6']
    df = pd.get_dummies(df, columns=continuous_features + categorical_features)
    return df


def preprocess_data_nypd(output_dir):
    """ Preprocesses nypd frisk data from original nypd_2012.csv
    Original source: https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page
    
    Args: 
      output_dir (string): output directory to which to save preprocessed data.
    """
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/nypd_2012.csv")
    df_nypd = pd.read_csv(data_path)

    # Only get rows where suspected crime is CPW.
    cpw_vals = []
    for col in df_nypd['crimsusp'].unique():
        if 'CPW' in str(col):
            cpw_vals.append(col)
        elif 'WEAPON' in str(col):
            cpw_vals.append(col)
        elif 'WEPON' in str(col):
            cpw_vals.append(col)
        elif 'WPN' in str(col):
            cpw_vals.append(col)
        elif 'CWP' in str(col):
            cpw_vals.append(col)
        elif 'C.P.W.' in str(col):
            cpw_vals.append(col)
    cpw_df = df_nypd[df_nypd['crimsusp'].isin(cpw_vals)]

    # Create weapon_found label.
    weapon_found = (cpw_df['pistol'] == 'Y') | (cpw_df['riflshot'] == 'Y') | (cpw_df['asltweap'] == 'Y') | (cpw_df['knifcuti'] == 'Y') | (cpw_df['machgun'] == 'Y') | (cpw_df['othrweap'] == 'Y')
 
    # Fix height feature.
    cpw_df['height'] = 12*cpw_df['ht_feet'] + cpw_df['ht_inch']
 
    # Map race codes to race strings.
    cpw_df['race'] = cpw_df['race'].replace(['B', 'W', 'P', 'Q', 'Z', 'A', 'U', 'I'], ['black', 'white', 'hisp', 'hisp', 'other', 'other', 'other', 'other'])

    # All columms to include
    features = ['weapon_found', 
    'pct', 
    'trhsloc', 
    'inout', 
    'sex', 
    'race', 
    'build', 
    'age', 
    'weight', 
    'height', 
    'perobs', 
    'offunif', 
    'radio', 
    'ac_rept',
    'ac_inves',
    'rf_vcrim',
    'rf_othsw',
    'ac_proxm',
    'rf_attir',
    'cs_objcs',
    'cs_descr',
    'cs_casng',
    'cs_lkout',
    'rf_vcact',
    'cs_cloth',
    'cs_drgtr',
    'ac_evasv',
    'ac_assoc',
    'cs_furtv',
    'rf_rfcmp',
    'ac_cgdir',
    'rf_verbl',
    'cs_vcrim',
    'cs_bulge',
    'cs_other',
    'ac_incid',
    'ac_time',
    'rf_knowl',
    'ac_stsnd',
    'ac_other',         
    'sb_hdobj',
    'sb_outln',
    'sb_admis',
    'sb_other',
    'rf_furt',
    'rf_bulg']
    cpw_df_filtered = cpw_df[features]

    # Only include black, white, hispanic
    cpw_df_filtered = cpw_df_filtered[cpw_df_filtered['race'].isin(['black', 'white', 'hisp'])]

    # Replace strings with binary labels
    cpw_df_filtered = cpw_df_filtered.replace('Y', 1)
    cpw_df_filtered = cpw_df_filtered.replace('N', 0)

    # Compute proxies from precinct.
    proxy_path = os.path.join(cur_dir, "data/nyc_2010pop_2020precincts.csv")
    df_proxy = pd.read_csv(proxy_path)
    precinct_proxy_map = {}
    for row_tup in df_proxy.iterrows():
        row = row_tup[1]
        precinct = row['precinct_2020']
        total = row['P0020001']
        num_white = row['P0020005']
        num_black = row['P0020006']
        num_hispanic = row['P0020002']
        proportions = [num_white/total, num_black/total, num_hispanic/total]
        precinct_proxy_map[precinct] = proportions

    # Normalize district_proxy_map
    precinct_proxy_map_normalized = {}
    for precinct, percents in precinct_proxy_map.items():
        total = sum(percents)
        norm = [percent/total for percent in percents]
        precinct_proxy_map_normalized[precinct] = norm
    def get_proxy(precinct):
        choices = ['white', 'black', 'hisp']
        norm = precinct_proxy_map_normalized[precinct]
        return np.random.choice(choices, 1, p=norm)[0]
    cpw_df_filtered['proxy_race'] = cpw_df_filtered['pct'].apply(get_proxy)

    # Binarize categorical columns
    cpw_df_filtered = pd.get_dummies(cpw_df_filtered, columns=['trhsloc', 'sex', 'race', 'proxy_race', 'build', 'inout'])

    numeric_filtered_df = cpw_df_filtered.apply(pd.to_numeric)
    numeric_filtered_df.to_csv(path_or_buf=os.path.join(output_dir, "nypd_processed.csv"), index=False)


def preprocess_data_boston(output_dir, min_officer_id_count=50):
    """ Preprocesses boston frisk data from original boston-police-department-fio.csv
    Original source: https://data.boston.gov/dataset/boston-police-department-fio/
    file: FIO RECORDS 2011 - 2015 (OLD RMS)
    link (as of 01/2020): https://data.boston.gov/dataset/4ebae674-28c1-4b9b-adc3-c04c99234a68/resource/c696738d-2625-4337-8c50-123c2a85fbad/download/boston-police-department-fio.csv

    Replaces NaNs with string "BLANK". Also converts label column into binary labels.

    Args: 
      output_dir (string): output directory to which to save preprocessed data.

    """
    def get_datetime_year(input_str):
        datetime_object = datetime.strptime(input_str, '%m/%d/%Y %I:%M:%S %p')
        return datetime_object.year
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/boston-police-department-fio.csv")
    data = pd.read_csv(data_path)
    all_columns = ["SEX", "FIO_DATE", "PRIORS", "COMPLEXION", "FIOFS_TYPE", "FIOFS_REASONS", "AGE_AT_FIO_CORRECTED", "DESCRIPTION", "DIST", "OFFICER_ID"]
    data_filtered = data[all_columns]
    data_filtered = data_filtered.fillna("BLANK")
    label_column = "FIOFS_TYPE"
    data_filtered[label_column] = np.where(data_filtered[label_column].str.contains("F|S", regex=True), 1, 0)
    data_filtered['FIO_YEAR'] = data_filtered['FIO_DATE'].apply(get_datetime_year)
    # Filter data to only include last two years (2014 and 2015)
    data_filtered = data_filtered[data_filtered['FIO_YEAR']>=2014]
    all_columns_final = ["SEX", "FIO_YEAR", "PRIORS", "COMPLEXION", "FIOFS_TYPE", "FIOFS_REASONS", "AGE_AT_FIO_CORRECTED", "DESCRIPTION", "DIST", "OFFICER_ID"]
    data_filtered = data_filtered[all_columns_final]
    # Filter dataframe by race to remove 'NO DATA ENTERED' and smaller race groups.
    filtered_df_race = data_filtered[(data_filtered.DESCRIPTION == 'B(Black)') | (data_filtered.DESCRIPTION == 'W(White)') | (data_filtered.DESCRIPTION == 'H(Hispanic)')]
    # Remove parentheses from race labels.
    def remove_parens(race):
        if race == 'B(Black)':
            return 'Black'
        elif race == 'W(White)':
            return 'White'
        else:
            return 'Hispanic'
    filtered_df_race["DESCRIPTION"] = filtered_df_race["DESCRIPTION"].apply(remove_parens)        

    # Add district proxies. for each district, racial composition is given by [white, black, hispanic].
    district_proxy_map = {
        'B2': [0.32, 0.53, 0.29], # 'B(Black)'
        'D4': [0.55, 0.12, 0.14], #'W(White)',
        'A1': [0.56, 0.04, 0.06], #'W(White)',
        'C11': [0.22, 0.44, 0.16], # 'B(Black)',
        'C6': [0.78, 0.05, 0.10], #'W(White)',
        'A7': [0.32, 0.02, 0.58], #'H(Hispanic)',
        'E13': [0.54, 0.12, 0.25], # 'W(White)'
        'B3': [0.06, 0.74, 0.16], #'B(Black)',
        'D14': [0.70, 0.04, 0.09], #'W(White)',
        'A15': [0.70, 0.09, 0.11], #'W(White)',
        'E18': [0.26, 0.45, 0.24], #'B(Black)',
        'E5': [0.73, 0.09, 0.09] #'W(White)'
    }
    # Filter out districts outside of the proxies.
    filtered_df_districts = filtered_df_race
    for district in filtered_df_race['DIST'].unique():
        if district not in district_proxy_map:
            filtered_df_districts = filtered_df_districts[filtered_df_districts['DIST'] != district]
    # Normalize district_proxy_map
    district_proxy_map_normalized = {}
    for district, percents in district_proxy_map.items():
        total = sum(percents)
        norm = [percent/total for percent in percents]
        district_proxy_map_normalized[district] = norm
    def get_proxy(district):
        choices = ['White', 'Black', 'Hispanic']
        norm = district_proxy_map_normalized[district]
        return np.random.choice(choices, 1, p=norm)[0]
    filtered_df_districts['PROXY_RACE'] = filtered_df_districts['DIST'].apply(get_proxy)

    # Filter out officer ids with count <= 300.
    officer_id_value_counts = filtered_df_districts['OFFICER_ID'].value_counts()
    def regroup_officer_id(id_value):
        if officer_id_value_counts[id_value] <= min_officer_id_count:
            return 0
        else:
            return id_value
    filtered_df_districts['OFFICER_ID'] = filtered_df_districts['OFFICER_ID'].apply(regroup_officer_id)
    filtered_df_districts.to_csv(path_or_buf=os.path.join(output_dir, "boston_processed.csv"), index=False)
    

    
def load_dataset_boston(binarize=True):
    """ Loads boston frisk dataset from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/boston_processed.csv")
    df = pd.read_csv(data_path)
    if binarize:
        categorical_columns = ["SEX", "FIO_YEAR", "PRIORS", "COMPLEXION", "FIOFS_REASONS", "DESCRIPTION", "DIST", "OFFICER_ID", "PROXY_RACE"]
        binarized_df = pd.get_dummies(df, columns=categorical_columns)
        return binarized_df
    else:
        return df


def load_dataset_nypd(downsample=True):
    """ Loads NYPD dataset from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/nypd_processed.csv")
    df = pd.read_csv(data_path)
    if downsample:
        df_neg = df[df['weapon_found'] == 0]
        df_pos = df[df['weapon_found'] == 1]
        df_neg_sampled = df_neg.sample(n=len(df_pos), random_state=100)
        # concatenate df_pos and df_neg
        df_sampled = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
        return df_sampled
    else:
        return df

    
def load_dataset_adult():
    """ Loads adult dataset from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir, "data/adult_processed.csv")
    df = pd.read_csv(data_path)
    return df


def preprocess_data_adult(output_dir):
    CATEGORICAL_COLUMNS = [
        'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country']
    CONTINUOUS_COLUMNS = [
        'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
    ]
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]
    LABEL_COLUMN = 'label'

    train_df_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=COLUMNS, skipinitialspace=True)
    test_df_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", names=COLUMNS, skipinitialspace=True, skiprows=1)

    train_df_raw[LABEL_COLUMN] = (train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    test_df_raw[LABEL_COLUMN] = (test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    # Preprocessing Features
    pd.options.mode.chained_assignment = None  # default='warn'

    # Functions for preprocessing categorical and continuous columns.
    def binarize_categorical_columns(input_train_df, input_test_df, categorical_columns=[]):

        def fix_columns(input_train_df, input_test_df):
            test_df_missing_cols = set(input_train_df.columns) - set(input_test_df.columns)
            for c in test_df_missing_cols:
                input_test_df[c] = 0
                train_df_missing_cols = set(input_test_df.columns) - set(input_train_df.columns)
            for c in train_df_missing_cols:
                input_train_df[c] = 0
                input_train_df = input_train_df[input_test_df.columns]
            return input_train_df, input_test_df

        # Binarize categorical columns.
        binarized_train_df = pd.get_dummies(input_train_df, columns=categorical_columns)
        binarized_test_df = pd.get_dummies(input_test_df, columns=categorical_columns)
        # Make sure the train and test dataframes have the same binarized columns.
        fixed_train_df, fixed_test_df = fix_columns(binarized_train_df, binarized_test_df)
        return fixed_train_df, fixed_test_df

    def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
        assert (num_quantiles is None or bins is None)
        if num_quantiles is not None:
            train_quantized, bins_quantized = pd.qcut(
              input_train_df[continuous_column_name],
              num_quantiles,
              retbins=True,
              labels=False)
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins_quantized, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins_quantized, labels=False)
        elif bins is not None:
            input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins, labels=False)

    # Filter out all columns except the ones specified.
    train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]]
    
    # Bucketize continuous columns.
    bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
    bucketize_continuous_column(train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
    bucketize_continuous_column(train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
    bucketize_continuous_column(train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
    bucketize_continuous_column(train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
    train_df, test_df = binarize_categorical_columns(train_df, test_df, categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
    full_df = train_df.append(test_df)
    full_df['race_Other_combined'] = full_df['race_Amer-Indian-Eskimo'] + full_df['race_Asian-Pac-Islander'] + full_df['race_Other']
    full_df.to_csv(path_or_buf=os.path.join(output_dir, "adult_processed.csv"), index=False)


def save_proxy_columns_adult(output_dir, proxy_noises=[0.1,0.2,0.3,0.4,0.5]):
    df = load_dataset_adult()
    protected_columns = ['race_White', 'race_Black', 'race_Other_combined']
    # Generate proxy groups.
    for noise in proxy_noises:
        df = generate_proxy_columns(df, protected_columns, noise_param=noise)
    df.to_csv(path_or_buf=os.path.join(output_dir, "adult_processed.csv"), index=False)


# Returns proxy column names given protected columns and noise param.
def get_proxy_column_names(protected_columns, noise_param):
    return ['PROXY_' + '%0.2f_' % noise_param + column_name for column_name in protected_columns]


def generate_proxy_columns(df, protected_columns, noise_param=1):
    """Generates proxy columns from binarized protected columns.

    Args: 
      df: pandas dataframe containing protected columns, where each protected 
        column contains values 0 or 1 indicating membership in a protected group.
      protected_columns: list of strings, column names of the protected columns.
      noise_param: float between 0 and 1. Fraction of examples for which the proxy 
        columns will differ from the protected columns.

    Returns:
      df_proxy: pandas dataframe containing the proxy columns.
      proxy_columns: names of the proxy columns.
    """
    proxy_columns = get_proxy_column_names(protected_columns, noise_param)
    num_datapoints = len(df)
    num_groups = len(protected_columns)
    noise_idx = random.sample(range(num_datapoints), int(noise_param * num_datapoints))
    proxy_groups = np.zeros((num_groups, num_datapoints))
    df_proxy = df.copy()
    for i in range(num_groups):
        df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
    for j in noise_idx:
        group_index = -1
        for i in range(num_groups):
            if df_proxy[proxy_columns[i]][j] == 1:
                df_proxy.at[j, proxy_columns[i]] = 0
                group_index = i
                allowed_new_groups = list(range(num_groups))
                allowed_new_groups.remove(group_index)
                new_group_index = random.choice(allowed_new_groups)  
                df_proxy.at[j, proxy_columns[new_group_index]] = 1
                break
        if group_index == -1:
            print('missing group information for datapoint ', j)
    return df_proxy


def compute_phats(df_proxy, proxy_columns):
    """Compute phat from the proxy columns for all groups.

    Args: 
      df_proxy: pandas dataframe containing proxy columns, where each proxy
        column contains values 0 or 1 indicating noisy membership in a protected group.
      proxy_columns: list of strings. Names of the proxy columns.
      
    Returns:
      phats: 2D nparray with float32 values. Shape is number of groups * number of datapoints
      Each row represents \hat{p}_j for group j. Sum of each row is approximatedly 1.
    """
    num_groups = len(proxy_columns)
    num_datapoints = len(df_proxy)
    phats = np.zeros((num_groups, num_datapoints), dtype=np.float32)
    for i in range(num_groups):
        group_name = proxy_columns[i]
        group_size = df_proxy[group_name].sum()
        proxy_col = np.array(df_proxy[group_name])
        for j in range(num_datapoints):
            if proxy_col[j] == 1:
                phats[i, j] = float(1/group_size)
    return phats

def train_val_test_split(df, train_percent, validate_percent, seed=88):
    """
    split the whole dataset as train set, validation set and test set according to the input percentage
    
    """
    np.random.seed(seed=seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test
