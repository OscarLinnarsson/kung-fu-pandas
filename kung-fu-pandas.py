
import numpy as np
import pandas as pd
import swifter


def rename_column(df, old_name, new_name):
    df.rename(columns = {f'{old_name}': f'{new_name}'})
    # new_col_names = [
    #     new_name if name == old_name else name
    #     for name in df.columns.tolist()
    # ]
    # df.columns = new_col_names
    return df


def remove_columns(df, cols):
    return df[df.columns.difference(cols)]


def keep_columns(df, cols):
    return df[cols]


def get_column_overlap(df1, df2):
    unique_col_df2 = df2.columns.difference(df1.columns)
    overlap_col_df2 = df2.columns.difference(unique_col_df2)
    return overlap_col_df2


def get_column_overlap_except(df1, df2, disregard):
    overlap = get_column_overlap(df1, df2)
    return overlap.difference(disregard)


def remove_rows_with_nan(df, cols=None, expressive_response=False):
    if cols is None:
        df_filtered = df.dropna(axis=0, how='any')
    else:
        df_filtered = df.dropna(axis=0, how='any', subset=cols)
    
    if expressive_response is False:
        return df_filtered

    nr_rows_before = df.shape[0]
    nr_rows_after = df_filtered.shape[0]
    nr_rows_removed = nr_rows_before - nr_rows_after
    frac_removed = round(nr_rows_removed / nr_rows_before * 100, 2)
    stat = {
        'nr_rows_before': nr_rows_before,
        'nr_rows_after': nr_rows_after,
        'nr_rows_removed': nr_rows_removed,
        'frac_removed': frac_removed,
    }
    return df_filtered, stat


def left_join(df1, df2, key_arr):
    overlap_exc_key = get_column_overlap_except(df1, df2, key_arr)
    tmp_df2 = remove_columns(df2, overlap_exc_key)
    return pd.merge(df1, tmp_df2, how='left', on=key_arr)


def inner_join(df1, df2, key_arr, name_df1=None, name_df2=None):
    df = outer_join(df1, df2, key_arr, name_df1=name_df1, name_df2=name_df2)
    tmp_overlap = get_column_overlap_except(df1, df2, key_arr)
    tmp_overlap_1 = (tmp_overlap + '_' + name_df1).tolist()
    tmp_overlap_2 = (tmp_overlap + '_' + name_df2).tolist()
    overlap = key_arr + tmp_overlap_1 + tmp_overlap_2
    return keep_columns(df, overlap)


def outer_join(df1, df2, key_arr, name_df1=None, name_df2=None):
    overlap_col2_exc_key = get_column_overlap_except(df1, df2, key_arr)
    df_m = pd.merge(df1, df2, how='outer', on=key_arr)

    if name_df1 is not None and name_df2 is not None:
        # Rename overlaping columns
        if name_df1 is not None:
            overlaping_cols = overlap_col2_exc_key + '_x'
            df_m.columns = [
                col if col not in overlaping_cols else col[:-1] + name_df2
                for col in df_m.columns
            ]
    else:
        # If no names are given to the left (df1) and right (df2) dataframes.
        # Merge overlaping columns giving priority to the df to the left.
        # Then remove the "right" ones
        for col in overlap_col2_exc_key:
            df_m[f'{col}_x'] = df_m[f'{col}_x'].fillna(df_m[f'{col}_y'])
        overlaping_cols_right = overlap_col2_exc_key + '_y'
        df_m = remove_columns(df_m, overlaping_cols_right)

        # Rename overlaping columns to be the same as before the merge
        overlaping_cols_left = overlap_col2_exc_key + '_x'
        df_m.columns = [
            col if col not in overlaping_cols_left else col.split('_')[0]
            for col in df_m.columns
        ]

    return df_m


def union(df1, df2):
    if (np.all(df1.columns == df2.columns)):
        return df1.append(df2).reset_index(drop=True)
    msg = "The two dataframes can not be merged " \
        + "as they do not have matching columns"
    raise Exception(msg)


def add_derived_column(df, name, func):
    df_copy = df.copy()
    if isinstance(name, str) and isinstance(func, list) is False:
        df_copy[name] = df_copy.swifter.apply(lambda x: func(x), axis=1)
        return df_copy
    return df


def filter_rows(df, func, return_inverse=False):
    df_f = df
    df_f = add_derived_column(df_f, 'pandas_power_tmp_col', func)
    df_f = df_f[df_f['pandas_power_tmp_col']]
    df_f = remove_columns(df_f, ['pandas_power_tmp_col'])
    if return_inverse is True:
        def inv_func(row):
            return func(row) is False
        df_inverse = filter_rows(df, inv_func)
        return df_f, df_inverse
    return df_f


def pivot(df, pivot, key=None, keys=None):
    if keys is None:
        if key is None:
            raise Exception('You need to provide either key or keys')
        keys = [key]
    arr1 = keys + [pivot]
    values = list(df.columns.difference(arr1))
    nr_rows_before = len(df.index)
    df = df.drop_duplicates(arr1)
    nr_rows_after = len(df.index)
    nr_rows_removed = nr_rows_before - nr_rows_after
    percent_left = round(100.0 * nr_rows_after / nr_rows_before)
    if percent_left <= 99.9:
        print(
            f'NR ROWS: removed={nr_rows_removed}   percent left={percent_left}'
        )
    df = df.pivot(index=keys, columns=pivot, values=values)
    return df


def add_aggregated_column(df, groupby_arr, col_name, agg_func_name):
    df_tmp = df
    df_tmp = df_tmp.groupby(groupby_arr)
    df_tmp = df_tmp.agg({f'{col_name}': f'{agg_func_name}'})
    df_tmp = df_tmp.reset_index()
    df_tmp = rename_column(df_tmp, col_name, f'{col_name}_{agg_func_name}')
    df = left_join(df, df_tmp, groupby_arr)
    return df


def check_column_existance(df, col_name):
    if col_name not in list(df.columns):
        raise Exception(f'The provided column "{col_name}" does not exist.')


def check_column_type(df, col_name, start_type_name):
    check_column_existance(df, col_name)
    col_type = df.dtypes[df.columns.index(col_name)].name
    if col_type.startswith(start_type_name) is False:
        raise Exception(
            f'The provided column "{col_name}" '
            + f'is not of the expected type "{start_type_name}" '
            + f'but of the type "{col_type}".'
        )


def get_datetime_column(df, attr_name):
    dt_columns = []
    cols = df.columns
    cols_type = df.dtypes
    for i in range(len(cols)):
        col = cols[i]
        col_type = cols_type[i].name
        if col_type.startswith('datetime'):
            dt_columns.append(col)
    if len(dt_columns) < 1:
        raise Exception('No datetime column found in the provided dataframe')
    
    if attr_name is None:
        if len(dt_columns) != 1:
            raise Exception(

            )
        attr_name = dt_columns[0]
    
    if attr_name not in cols:
        raise Exception(

        )
    
    if attr_name not in dt_columns:
        attr_name_type = cols_type[cols.index(attr_name)]
        raise Exception(
            
        )

    return attr_name


def add_year(df, name='year', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.year
    return df


def add_month(df, name='month', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.month
    return df


def add_day(df, name='day', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.day
    return df


def add_hour(df, name='hour', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.hour
    return df


def add_week(df, name='week', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.week
    return df


def add_weekday(df, name='weekday', attr_name=None):
    attr_name = get_datetime_column(df, attr_name)
    df[name] = df[attr_name].dt.dayofweek
    return df

