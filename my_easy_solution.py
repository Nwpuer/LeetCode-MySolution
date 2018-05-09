#这个文件做总的流程和数据处理的工作，以后可以将其分出去
import numpy as np 
import pandas as pd  
import gc
from FE import extract_features
from lgbm import lgb_fun

def Do(frm, to):
    dtypes = {
        'ip':'uint32',
        'app':'uint16',
        'device':'uint16',
        'os':'uint16',
        'channel':'uint16',
        'is_attributed':'uint8',
        'click_id':'uint32'
    }

    print('loading train data...', frm, to)
    train_df = pd.read_csv('../input/train.csv', parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','os','device','channel','click_time','is_attributed'])

    print('loading test data')
    test_df = pd.read_csv('../input/test.csv', parse_dates=['click_time'], dtype=dtypes)

    len_train = len(train_df)
    train_df = train_df.append(test_df)
    del test_df
    gc.collect()

    train_df['is_attributed'].fillna(-1, inplace=True)
    train_df['is_attributed'] = train_df['is_attributed'].astype('uint8',copy=False)
    train_df['click_id'].fillna(-1, inplace=True)
    train_df['click_id'] = train_df['click_id'].astype('uint32', copy=False)

    print('start extracting features...')
    train_df['day'] = pd.to_datetime(train_df['click_time']).dt.day.astype('uint8',copy=False)
    train_df['hour'] = pd.to_datetime(train_df['click_time']).dt.hour.astype('uint8',copy=False)
    gc.collect()

    train_df,features = extract_features(train_df)
    gc.collect()

    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    predictors = list(set(features) + set(categorical))
    print('predictors:', predictors)
    print('_____________________')
    print('categorical:', categorical)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train - val_size):len_train]
    train_df = train_df[:(len_train - val_size)]
    gc.collect()

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id']

    print('model training...')
    predictions = lgb_fun(
                            train_df, 
                            val_df, 
                            test_df,
                            predictors, 
                            target, 
                            categorical
    )
    del train_df
    del val_df
    del test_df
    gc.collect()

    sub['is_attributed'] = predictions.values
    print('writing')
    sub.to_csv('submission.csv', index=False, float_format='%.9f')
    del sub
    gc.collect()
    print('done!')





    











#Main function--------------------------------------------------------
if __name__ == '__main__':
    nrows=184903891-1 # the first line is columns' name
    nchunk=25000 # 【The more the better】
    val_size=2500
    frm=nrows-75000000
    to = frm + nchunk

    Do(frm, to)
