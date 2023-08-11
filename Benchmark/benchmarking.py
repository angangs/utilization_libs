import pandas as pd
from DEA import DEABenchmarking


def write_dataframe(df, filename, sheet_name=None):
    extension = filename.split(".")[1]
    if extension == "csv":
        df.to_csv(filename, index=False, encoding='utf-8-sig', sep=';')
    else:
        df.to_excel(filename, sheet_name=sheet_name, index=False)
    return


def read_dataframe(filename, sheetname=None, columns=None, index_col=None):
    extension = filename.split(".")[1]
    if extension == "csv":
        # Read dataframe
        df = pd.read_csv(filename, index_col=index_col, encoding='utf-8-sig', sep=';')
        if columns is None:
            return df
        else:
            return df[columns]
    else:
        # Read dataframe
        df = pd.read_excel(filename, sheet_name=sheetname, index_col=index_col)
        if columns is None:
            return df
        else:
            return df[columns]


def dea_benchmarking(df_input, df_target, input_names, target_column_name, id_column_name, label_column_name):
    df_data = pd.concat([df_input, df_target], axis=1)
    dea_list = []
    input_names.remove(label_column_name)
    input_names.remove(id_column_name)
    for label in df_input[label_column_name].unique().tolist():
        df_data_part = df_data[df_data[label_column_name] == label]
        fixed_input_names = []
        for col in input_names:
            if df_data_part[col].corr(df_data_part[target_column_name]) > 0.15:
                fixed_input_names.append(col)
        print('DEA for label {} with features: {}'.format(label, fixed_input_names))
        dea = DEABenchmarking(df_data_part, fixed_input_names, df_target.columns.tolist(), id_column_name)
        dea.dea_reducted_sample()
        # dea.dea_convex_hull()
        dea_list.append(dea.efficient_level_input_output)
    dea_df = pd.concat(dea_list, axis=0)
    dea_df = dea_df.reset_index()
    return dea_df[['index', 'Theta']]
