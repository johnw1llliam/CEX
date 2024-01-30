import pandas as pd
import numpy as np
import matplotlib as plt    
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Exploring the dataframe properties
def explore(df):
    print(f"{df.dtypes}\n")
    print(f"Dimension: {df.shape[0]} x {df.shape[1]}\n")
    
    datatype_counts = df.dtypes.value_counts()
    for dtype, count in datatype_counts.items():
        print(f"{dtype}: {count} columns")

# Get column category items frequency (text)
def col_category_text(df):
    for col in df.select_dtypes("object").columns:
        print(df[col].value_counts())
        print("\n")

# Get column category items frequency (graph)
def col_category_graph(df):
    for col in df.select_dtypes("object").columns:
        value_counts = df[col].value_counts()
        pyplot.figure(figsize=(10, 6))  
        sns.barplot(x=value_counts.index, y=value_counts.values)
        pyplot.title(f'Bar Chart for {col}')
        pyplot.xlabel(col)
        pyplot.ylabel('Count')
        pyplot.xticks(rotation=45)  
        pyplot.tight_layout()  
        pyplot.show()
        print("\n")

# Checking for nulls (text)
def null_check_text(df):
    null = df.isnull().sum()
    for i in range(len(df.columns)):
        print(f"{df.columns[i]}: {null[i]} ({(null[i]/len(df))*100}%)")
    total_cells = np.prod(df.shape)
    total_missing = null.sum()
    print(f"\nTotal missing values: {total_missing} ({(total_missing/total_cells) * 100}%)\n")

# Checking for nulls (graph)
def null_check_graph(df):
    col_name = []
    null_num = []
    percentage = []
    col_type = []

    null = df.isnull().sum()
    for i in range(len(df.columns)):
        if null[i] != 0:
            col_name.append(df.columns[i])
            null_num.append(null[i])
            percentage.append((null[i]/len(df))*100)
            col_type.append(df[df.columns[i]].dtypes)
    total_cells = np.prod(df.shape)
    total_missing = null.sum()
    col_name.append("total_missing")
    null_num.append(total_missing)
    percentage.append((total_missing/total_cells) * 100)
    col_type.append("total_column")
    null_dict = {
        "col_name" : col_name,
        "null_num" : null_num,
        "percentage" : percentage,
        "col_type" : col_type
    }
    null_df = pd.DataFrame.from_dict(null_dict)
    ax = sns.barplot(x="col_name", y="percentage", hue="col_type", data=null_df)
    pyplot.xticks(rotation=90)
    pyplot.show()

# Finding unique null columns for both training and testing data
def find_unique_null_columns(train_df, test_df):
    null_columns_train = train_df.columns[train_df.isnull().any()]
    null_columns_test = test_df.columns[test_df.isnull().any()]
    
    unique_null_columns_train = null_columns_train.difference(null_columns_test)
    unique_null_columns_test = null_columns_test.difference(null_columns_train)
    
    result_df = pd.DataFrame(columns=["Column_Name", "Exists_In"])
    
    result_df["Column_Name"] = unique_null_columns_train.union(unique_null_columns_test)
    result_df["Exists_In"] = ["Train" if col in unique_null_columns_train else "Test" for col in result_df["Column_Name"]]
    
    return result_df

# Finding different cardinality columns in training and test dataset
def different_cardinality_columns(train_df, test_df):
    differing_columns = []

    for column in train_df.columns:
        if column in test_df.columns:
            if train_df[column].dtype == 'object':
                unique_values_train = train_df[column].nunique()
                unique_values_test = test_df[column].nunique()

                if unique_values_train != unique_values_test:
                    differing_columns.append((column, "train" if unique_values_train > unique_values_test else "test", unique_values_train, unique_values_test))

    if differing_columns:
        diff_df = pd.DataFrame(differing_columns, columns=["Column", "Origin", "Cnt Cardinality in Train", "Cnt Cardinality in Test"])
        return diff_df
    else:
        return None

# Check for duplicate
def duplicate_check(df):
    print(df[df.duplicated()])

# Doing arithmetic operation in a column
def arithmetic_one_column(df, result_col_name, operation_col_name, operation_type, number):
    if operation_type == "add":
        df[result_col_name] = df[operation_col_name].add(number)
    elif operation_type == "sub":
        df[result_col_name] = df[operation_col_name].sub(number)
    elif operation_type == "mul":
        df[result_col_name] = df[operation_col_name].mul(number)
    elif operation_type == "div":
        df[result_col_name] = df[operation_col_name].div(number)
    else:
        print("Pick a correct operation choice")
    return df

# Doing arithmetic operation for 2 columns  
def arithmetic_two_column(df, result_col_name, operation_col_name_1, operation_col_name_2, operation_type):
    if operation_type == "add":
        df[result_col_name] = df[operation_col_name_1] + df[operation_col_name_2]
    elif operation_type == "sub":
        df[result_col_name] = df[operation_col_name_1] - df[operation_col_name_2]
    elif operation_type == "mul":
        df[result_col_name] = df[operation_col_name_1] * df[operation_col_name_2]
    elif operation_type == "div":
        df[result_col_name] = df[operation_col_name_1] / df[operation_col_name_2]
    else:
        print("Pick a correct operation choice")
    return df

# Changing the date format using strftime
def change_date_format(df, result_col_name, date_col, date_format):
    df[result_col_name] = pd.to_datetime(date_col)
    df[result_col_name] = df[result_col_name].dt.strftime(date_format)
    return df

# Taking a date element from a column
def taking_date_element(df, new_col_name, date_col_name, date_col, element_taken):
    df[date_col_name] = pd.to_datetime(date_col)
    if element_taken == "day":
        df[new_col_name] = df[date_col_name].dt.day
    elif element_taken == "dow":
        df[new_col_name] = df[date_col_name].dt.dayofweek
    elif element_taken == "month":
        df[new_col_name] = df[date_col_name].dt.month
    elif element_taken == "year":
        df[new_col_name] = df[date_col_name].dt.year
    else:
        print("Pick a correct element")
    return df

# Casting data type from a column
def casting(df, new_col_name, target_col, operation_type):
    if operation_type == "str_to_int":
        df[new_col_name] = df[target_col].astype(int)
    elif operation_type == "str_to_datetime":
        user_format = input("Input format: ")
        df[new_col_name] = pd.to_datetime(df[target_col], format=user_format)
    elif operation_type == "int_to_str":
        df[new_col_name] = df[target_col].apply(str)
    elif operation_type == "str_to_bool":
        df = df.replace({'True': True, 'False': False})
    else:
        print("Pick a correct operation type!")
        return
    return df
    
# Concatenate string in 2 columns
def combine_str_col(df, new_col_name, col_name_1, col_name_2, sep=" "):
    df[new_col_name] = df[col_name_1] + sep + df[col_name_2]
    return df

# Doing something for incomplete and blank (null) data
def nan(df, choice, col_name = None, new_col_name = None, fill = None):
    if choice == "fill_col":
        df[new_col_name] = df[col_name].fillna(fill)
    elif choice == "fill_all":
        df = df.fillna(fill)
    else:
        df = df.dropna(subset=[col_name])
    return df

# Merging 2 dataframe
def merge(df1, df2, axis = 1, join = "inner"):
    result = pd.concat([df1, df2], axis=axis, join=join)
    return result

# Adding new column
def add_column(df, new_column_name, column):
    df[new_column_name] = column
    return df

# delete a column or multiple columns
def delete_column(df, col_list):
    df = df.drop(col_list, axis = 1)
    return df

# Removing missing values
def remove_na(df, subset_in):
    df = df.dropna(subset = [subset_in], axis = 0)
    df.reset_index(drop=True, inplace=True)
    return df

# Replacing NaN with average
def replace_avg(df, col):
    avg = df[col].astype("float").mean(axis=0)
    df[col].replace(np.nan, avg, inplace=True)
    return df

# Replacing NaN with median
def replace_median(df, col):
    median = df[col].median()
    df[col].replace(np.nan, median, inplace=True)
    return df

# Replacing NaN with mode (categorical)
def replace_mode(df, col):
    mode = df[col].value_counts().idxmax()
    df[col].replace(np.nan, mode, inplace = True)
    return df

# Perform one hot encoding
def one_hot(df, col_list):
    df = pd.get_dummies(df, columns=col_list)
    return df

# Perform label encoding
def label_enc(df, col_list):
    df[col_list] = df[col_list].apply(LabelEncoder().fit_transform)
    return df   

# Perform target encoding
def target_encoding(train_df, test_df, cols, target_name):
    for column in cols:
        data = train_df.groupby(column)[target_name].mean()
        for value in data.index:
            train_df[column] = train_df[column].replace({value:data[value]})
            test_df[column] = test_df[column].replace({value:data[value]})
    return train_df, test_df

# set a column to become an index (UNTESTED)
def set_index(df, col_name):
    df.set_index(col_name, inplace=True)
    df.index.name = None

# renaming a column
def rename_col(df, change_dict):
    df.rename(columns = change_dict, inplace = True)

# Filtering rows with piece of string in it
def filter_rows(df, col, word):
    search = [word]
    result = df[df.eval(col).str.contains('|'.join(search))]
    return result   
    
# Delete rows that contain certain word
def del_row_c(df, col_name, word):
    df = df.loc[df[col_name] != word]
    return df

# Delete rows that not contain certain word
def del_row_nc(df, col_name, word):
    df = df.loc[df[col_name] == word]
    return df

# Creating Dummy Variable / Indicator Variable
def dummy_var(df, col, cat1, cat2, result_name1, result_name2):
    dummy = pd.get_dummies(df[col])
    dummy.rename(columns={cat1:result_name1, cat2:result_name2}, inplace=True)
    df = pd.concat([df, dummy], axis=1)
    return df

# Binning
def binning(df, col, sult_col):
    plt.pyplot.hist(df[col])
    plt.pyplot.xlabel(col)
    plt.pyplot.ylabel("count")
    plt.pyplot.title(f"{col} bins")
    pyplot.show()
    bin_num = input("How many bins: ")
    bins = np.linspace(min(df[col]), max(df[col]), bin_num)
    group_names = []
    for i in range(bin_num):
        bin = input(f"Group Name {i}: ")
        group_names.append(bin)
    df[sult_col] = pd.cut(df[col], bins, labels=group_names, include_lowest=True )
    plt.pyplot.hist(df[col], bins = bin_num)
    plt.pyplot.xlabel(col)
    plt.pyplot.ylabel("count")
    plt.pyplot.title(f"{col} bins")
    return df

# Replacing string/character for whole dataset
def replace_str(df, str1, str2):
    df = df.replace(str1, str2, regex=True)
    return df

# Check duplicate data and remove them
def remove_duplicate(df):
    bool_duplicate_series = df.duplicated()
    return(df[~bool_duplicate_series])

# Check for old data that less than certain time and remove them
def rm_old_rows(df, time_column, duration_inc):
    print(f"\nBefore Cleaning: {df.shape}\n")
    df = df.loc[df[time_column] >= duration_inc]
    print(f"\nAfter Cleaning: {df.shape}\n")
    return df

# Check for incorrect data (different from the parameter) on certain column 
def inc_data(df, col_name, choice, dropped_data = None, old_data = None, new_data = None):
    occur = df.groupby([col_name]).size()
    print(f"\n{occur}")
    print(f"\n{df.shape}\n")

    if choice == "delete":
        df = df[~df.isin([dropped_data]).any(axis=1)]

        occur = df.groupby([col_name]).size()
        print(f"\n{occur}")
        print(f"\n{df.shape}\n")
    elif choice == "replace":
        df[col_name] = df[col_name].replace(to_replace=old_data, value=new_data)
        occur = df.groupby([col_name]).size()
        print(f"\n{occur}")
    else:
        print("Wrong Choice")
    return df

# Check for inconsistent data (whitespace, typo, and data types) on certain column
def rm_inconsistencies(df, col_name, choice, old_value = None, new_value = None, new_col_name = None, op = None):
    occur = df.groupby([col_name]).size()
    print(occur)
    print(df.shape)

    if choice == "typo":
        # Typo
        df[col_name] = df[col_name].replace(to_replace=old_value, value=new_value)
        occur = df.groupby([col_name]).size()
        print(occur)
        # Whitespace
    elif choice == 'whitespace':
        df[col_name] = df[col_name].str.strip()
        # Data Types
    elif choice == "data_type":
        casting(df, new_col_name, col_name, op)
    else:
        print("Wrong Choice")
    return df

# Create a boxplot for categorical data
def boxplot_cat(col_list, df, target_col_name):
    plt.figure(figsize=(22,12))
    for index,column in enumerate(col_list):
        plt.subplot(2,4,index+1)
        sns.boxplot(x=column, y=target_col_name, data=df)
        plt.title(f"{column} vs {target_col_name}",fontweight="black",pad=10,size=20)
        plt.xticks(rotation=90)
        plt.tight_layout()
    
# Creating correlation matrix
def corr_matrix(df, target_col_name, annot):
    correlation = df.corrwith(df[target_col_name])
    correlation_df = pd.DataFrame(correlation, columns=[target_col_name])

    if annot:
        sns.heatmap(
        correlation_df.transpose(),
        vmin=correlation.values.min(),
        vmax=1,
        square=True,
        cmap="YlGnBu",
        linewidths=0.1,
        annot=True,
        annot_kws={"fontsize":8}
        )
        plt.show()
    else:
        sns.heatmap(correlation_df.transpose(), cmap="YlGnBu")  
        plt.show()

# Creating histogram for checking skewness and outliers
def skewness_outlier(df, cols):
    for col_name in cols:
        plt.figure(figsize=(13,6))

        plt.subplot(1,2,1)
        sns.histplot(df[col_name],color="purple",kde=True)
        plt.title(f"{col_name} Distribution Plot",fontweight="black",pad=20,size=18)

        plt.subplot(1,2,2)
        sns.boxplot(df[col_name],color="purple")
        plt.title(f"{col_name} Outliers Detection",fontweight="black",pad=20,size=18)
        plt.tight_layout()
        plt.show()
        print("\n")

# Creating bar plot to check skewness
def skewness_bar(df, cols):
    skewness = df[cols].skew().sort_values()

    plt.figure(figsize=(14,6))
    sns.barplot(x=skewness.index, y=skewness, palette=sns.color_palette("Reds",19))
    for i, v in enumerate(skewness):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom",size=15,fontweight="black")

    plt.ylabel("Skewness")
    plt.xlabel("Columns")
    plt.xticks(rotation=90)
    plt.title("Skewness of Numerical Columns",fontweight="black",size=20,pad=10)
    plt.tight_layout()
    plt.show()


