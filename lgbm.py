import lightgbm as lgb 
import matplotlib.pyplot as plt
import time
import gc

def lgb_modelfit_nocv(train_df, val_df, predictors, target, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None, metrics='auc'):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2, # 【consider using 0.1】
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth), default=31
        'max_depth': 3,  # -1 means no limit, default=-1
        'min_data_per_leaf': 100,  # alias=min_data_per_leaf , min_data, min_child_samples, default=20
        'max_bin': 100,  # Number of bucketed bin for feature values,default=255
        'subsample': 0.7,  # Subsample ratio of the training instance.default=1.0, alias=bagging_fraction
        'subsample_freq': 1,  # k means will perform bagging at every k iteration, <=0 means no enable,alias=bagging_freq,default=0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.alias:feature_fraction
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf),default=1e-3,Like min_data_in_leaf, it can be used to deal with over-fitting
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4, # should be equal to REAL cores:http://xgboost.readthedocs.io/en/latest/how_to/external_memory.html
        'verbose': 0
        #         'device': 'gpu',
#         'gpu_platform_id':1
        # gpu_use_dp, default=false,set to true to use double precision math on GPU (default using single precision)
#         'gpu_device_id': 2 #default=-1,default value is -1, means the default device in the selected platform
        # 'random_state':666 [LightGBM] [Warning] Unknown parameter: random_state
        # 'feature_fraction_seed': 666,
        # 'bagging_seed': 666, # alias=bagging_fraction_seed
        # 'data_random_seed': 666 # random seed for data partition in parallel learning (not include feature parallel)
    }

    print('convert the train data and validation data into dataset...')
    train_data_v1 = lgb.Dataset(train_df[predictors].values, label=train_df[target].values, feature_name=predictors, categorical_feature=categorical_features)
    del train_df
    gc.collect()
    train_data_v1.save_binary('train_v1.bin')
    train_data = lgb.Dataset('train_v1.bin', feature_name=predictors, categorical_feature=categorical_features)
    del train_data_v1
    gc.collect()

    valid_data_v1 = lgb.Dataset(val_df[predictors].values, label=val_df[target].values, feature_name=predictors,  categorical_feature=categorical_features)
    del val_df
    gc.collect()
    valid_data_v1.save_binary('valid_v1.bin')
    valid_data = lgb.Dataset('valid_v1.bin', feature_name=predictors, categorical_feature=categorical_features)
    del valid_data_v1
    gc.collect()

    evals_results={}
    bst = lgb.train(lgb_params, 
                     train_data, 
                     valid_sets=[valid_data], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval)
    
    del train_data, valid_data
    gc.collect()
    print('model report:')
    print('best iteration:', bst.best_iteration)
    print(metrics+':', evals_results['valid'][metrics][bst.best_iteration-1])

    return bst, bst.best_iteration






def lgb_fun(train_df, val_df, test_df, predictors, target, categorical):
    print('training...')
    start_time = time.time()

    bst,best_iteration = lgb_modelfit_nocv(
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            early_stopping_rounds=50, 
                            verbose_eval=True, 
                            num_boost_round=2000, 
                            categorical_features=categorical)
    
    print('model training time:{}'.format(time.time() - start_time))
    
    print('plot feature importance')
    lgb.plot_importance(bst)
    plt.gcf().savefig('feature_importance_split.png')
    lgb.plot_importance(bst, importance_type = 'gain')
    plt.gcf().savefig('feature_importance_gain.png')

    print('predicting...')
    predictions = bst.predict(test_df[predictors], num_iteration=best_iteration)

    return predictions

