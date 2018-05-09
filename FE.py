import pandas as pd 
import numpy as np 
import gc

features = []

def do_countuniq(df, groupby, select, agg_type='uint32', show_max=False, show=True):
    if show:
        print('do_countuniq', select, 'by', groupby, '...')
    agg_name = '{}_{}_{}'.format('_'.join(groupby), 'countuniq', select)
    gp = df[list(set(groupby+[select]))].groupby(groupby)[select].nunique().reset_index().rename(columns={select:agg_name})
    if show_max:
        print(agg_name, 'max value', gp[agg_name].max())
    df = df.merge(gp, on=groupby, how='left', copy=False)
    del gp
    gc.collect()
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    features.append(agg_name)
    return df

def do_count(df, groupby, agg_type='uint32', show_max=False, show=True):
    if show:
        print('do_count by', groupby, '...')
    agg_name = '{}_{}'.format('_'.join(groupby), 'count')
    gp = df[groupby].groupby(groupby).size().rename(agg_name).to_frame().reset_index()
    if show_max:
        print(agg_name, 'max value', gp[agg_name].max())
    df = df.merge(gp, on=groupby, how='left', copy=False)
    del gp
    gc.collect()
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    features.append(agg_name)
    return df

def do_cumcount(df, groupby, agg_type='uint32', show_max=False, show=True):
    if show:
        print('do_cumcount by', groupby, '...')
    agg_name = '{}_{}'.format('_'.join(groupby), 'cumcount')
    gp = df[groupby].groupby(groupby).cumcount()    #different here.Am I right?
    df[agg_name] = gp.values
    if show_max:
        print(agg_name, 'max_value', gp.max())
    del gp
    gc.collect()
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    features.append(agg_name)
    return df

def do_mean(df, groupby, select, agg_type='float32', show_max=False, show=True):
    if show:
        print('do_mean', select, 'by', groupby, '...')
    agg_name = '{}_{}_{}'.format('_'.join(groupby), 'mean', select)
    gp = df[list(set(groupby+[select]))].groupby(groupby)[select].mean().reset_index().rename(columns={select:agg_name})
    if show_max:
        print(agg_name, 'max value', gp[agg_name].max())
    df = df.merge(gp, on=groupby, how='left', copy=False)
    del gp
    gc.collect()
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    features.append(agg_name)
    return df

def do_var(df, groupby, select, agg_type='float32', show_max=False, show=True):
    if show:
        print('do_var', select, 'by', groupby, '...')
    agg_name = '{}_{}_{}'.format('_'.join(groupby), 'var', select)
    gp = df[list(set(groupby+[select]))].groupby(groupby)[select].var().reset_index().rename(columns={select:agg_name})
    if show_max:
        print(agg_name, 'max value', gp[agg_name].max())
    df = df.merge(gp, on=groupby, how='left', copy=False)
    del gp
    gc.collect()
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    features.append(agg_name)
    return df

def do_nextClick(df, groupby, type='uint32', show_max=False, show=True):
    if show:
        print('do_nextClick by', groupby, '...')
    feature_name = '{}_{}'.format('_'.join(groupby), 'nextClick')
    all_features = list(set(groupby + ['click_time']))
    gp = df[all_features].groupby(groupby).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    if show_max:
        print(feature_name, 'max value', gp.max())
    df[feature_name] = gp.values
    del gp
    gc.collect()
    df[feature_name] = df[feature_name].astype(type, copy=False)
    features.append(feature_name)
    return df



def extract_features(train_df):
    train_df = do_countuniq(train_df, ['ip'], 'channel', 'uint8'); gc.collect()
    train_df = do_countuniq(train_df, ['ip', 'day'], 'hour', 'uint8'); gc.collect()
    train_df = do_countuniq(train_df, ['ip'], 'app', 'uint16'); gc.collect()
    train_df = do_countuniq(train_df, ['ip', 'app'], 'os', 'uint8'); gc.collect()
    train_df = do_countuniq(train_df, ['ip'], 'device', 'uint16'); gc.collect()
    train_df = do_countuniq(train_df, ['app'], 'channel', 'uint8'); gc.collect()
    train_df = do_countuniq(train_df, ['ip', 'device', 'os'], 'app', 'uint8'); gc.collect()
    train_df = do_countuniq(train_df, ['ip', 'device', 'os'], 'channel', 'uint8'); gc.collect()

    train_df = do_cumcount(train_df, ['ip', 'os'], 'uint32'); gc.collect()
    train_df = do_cumcount(train_df, ['ip', 'device', 'os', 'app'], 'uint32'); gc.collect() #意思应该是点击的cumcount越小，该点击是欺诈的概率比较小，越大则概率越大，因为人不会这么无聊点着玩

    train_df = do_count(train_df, ['ip', 'app', 'channel'], 'uint32'); gc.collect()
    train_df = do_count(train_df, ['ip', 'device', 'os', 'app'], 'uint32'); gc.collect()
    train_df = do_count(train_df, ['ip', 'day', 'hour'], 'uint16'); gc.collect()
    train_df = do_count(train_df, ['ip', 'app'], 'uint32'); gc.collect()
    train_df = do_count(train_df, ['ip', 'app', 'os'], 'uint16'); gc.collect()
    
    train_df = do_var(train_df, ['ip', 'day', 'channel'], 'hour', 'float32'); gc.collect()
    train_df = do_var(train_df, ['ip', 'app', 'os'], 'hour', 'float32'); gc.collect()
    train_df = do_var(train_df, ['ip', 'app', 'channel'], 'day', 'float32'); gc.collect()

    train_df = do_mean(train_df, ['ip', 'app', 'channel'], 'hour', 'float32'); gc.collect()

    train_df = do_nextClick(train_df, ['ip'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'app'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'channel'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'os'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'app', 'device', 'os', 'channel'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'os', 'device'], 'uint32'); gc.collect()
    train_df = do_nextClick(train_df, ['ip', 'os', 'device', 'app'], 'uint32'); gc.collect()
    train_df.drop(['click_time','day'], axis=1, inplace=True); gc.collect()


    return train_df, features