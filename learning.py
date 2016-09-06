from features import *
from utils import *
import dictionaries


# --------------------------
# ---     Loadings       ---
# --------------------------

def learning_load_all_dfs(use_hl=False, use_both_for_obj=False):
    if use_hl:
        all_features_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/all_features_hl.pickle')
        df_moments = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/moments_features_hl.pickle')
        df_quantized = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/quantized_features_hl.pickle')
        df_dynamic = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/dynamic_features_hl.pickle')
        df_misc = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/misc_features_hl.pickle')

    else:
        all_features_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/all_features.pickle')
        df_moments = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/moments_features.pickle')
        df_quantized = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/quantized_features.pickle')
        df_dynamic = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/dynamic_features.pickle')
        df_misc = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/features/misc_features.pickle')

    if use_both_for_obj:
        objective_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/objective_both.pickle')
    else:
        objective_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/objective.pickle')

    majority_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/majority.pickle')
    ratings_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/ratings.pickle')
    big5_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/big5.pickle')
    raw_df = load_pickle_to_df(dictionaries.PICKLES_FOLDER + '/raw.pickle')

    print(" > Done Loading!\n")

    return all_features_df, df_moments, df_quantized, df_dynamic, df_misc, objective_df, ratings_df, big5_df, raw_df, majority_df


# --------------------------
# --  Feature Selection   --
# --------------------------

def pca(pca_each_axis, df_to_pca, df_not_to_pca, fs_n_components):
    feat_arr = df_not_to_pca

    if pca_each_axis:
        for df in df_to_pca:
            feat_arr.append(pd.DataFrame(PCA(n_components=fs_n_components).fit_transform(df), index=df.index))
        feat_df = pd.concat(feat_arr, axis=1)
    else:
        all_feat_df = pd.concat(df_to_pca, axis=1)
        feat_df = pd.DataFrame(PCA(n_components=fs_n_components).fit_transform(all_feat_df), index=all_feat_df.index)

    return feat_df

def feature_selection(X, fs_model_name, n_components=3):

    if fs_model_name == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        narrowed_x = pca.fit_transform(X)

    if fs_model_name == 'kernel_pca':
        pass
    #     todo

    if fs_model_name == 'sparse_pca':
        pass
    #     todo

    if fs_model_name == 'random_projections':
        pass
    #     todo

    if fs_model_name == 'none':
        narrowed_x = X

    return narrowed_x

# --------------------------
# ---      Learning      ---
# --------------------------

def run(X, Y, learning_model_name, cv_model_name, is_scaled=True, is_normalized=False,
        f=None, use_single_predicted_Y_foreach_clip=False, is_second_learner=False,
        corr_method='normal'):
    # this function learns X&Y using learning_model and evaluates it using cv_model

    all_corr_grades = []

    if is_scaled:
        from sklearn import preprocessing
        X = preprocessing.scale(X)

    Y_predicted_arr = []
    Y_test_arr = []

    # train-set and test-set are identical (and equal to the entire set)
    if cv_model_name == 'none':
        clf = run_learning(X, Y, learning_model_name)
        return calculate_corr(flatten_list(Y), flatten_list(clf.predict(X)), method=corr_method)

    else:
        cv_model_idx_list = split_data(len(Y), cv_model_name)
        for train_idx, test_idx in cv_model_idx_list:
            # Y_predicted_arr = []
            # Y_test_arr = []

            # split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # learn and append y-predicted AND y-true values
            clf = run_learning(X_train, Y_train, learning_model_name, is_normalized)
            Y_predicted = clf.predict(X_test)

            # calculate segment size
            seg_size = int(len(X)/(NUM_CLIPS*len(dictionaries.SUBJECTS_IDS)))
            if seg_size < 1:    # when computing a model for each subj, len(x) is as long as one subject's data
                seg_size = int(len(X)/NUM_CLIPS)

            if is_second_learner:
                second_learner_X = []
                first_learner_predicted_Y = clf.predict(X_train)              # fit-predict
                chunks_iter = grouper(first_learner_predicted_Y, seg_size)    # iterator of chunks of seg_size
                for clip in chunks_iter:
                    second_learner_X.append(
                        (np.var(clip), np.mean(clip)))                        # create X for 2nd learner

                clf2 = run_learning(second_learner_X, Y_train[0::seg_size],
                                    learning_model_name, is_normalized)        # run 2nd learner

                # -- at this point the 2nd model was computed --

                chunks_iter = grouper(Y_predicted, seg_size)                  # iterator of chunks of seg_size
                for clip in chunks_iter:                                      # learn each clip
                    Y_predicted_arr.append(clf2.predict((np.var(clip), np.mean(clip))))
                Y_test_arr.extend(Y_test[0::seg_size])

                # print(SUBJECTS_IDS[int(test_idx[0]/54)])

            else:
                # compute a single grade for each clip (a clip is the size of seg_size)
                if use_single_predicted_Y_foreach_clip:
                    chunks_iter = grouper(Y_predicted, seg_size)    # iterator of chunks of seg_size
                    for clip in chunks_iter:
                        Y_predicted_arr.append(np.median(clip))
                    Y_test_arr.extend(Y_test[0::seg_size])
                else:
                    Y_predicted_arr.append(Y_predicted)     # add predicted y to predicted array
                    Y_test_arr.append(Y_test)               # add actual    y to actual    array

            # all_corr_grades.append(calculate_corr(flatten_list(Y_test_arr), flatten_list(Y_predicted_arr), method=corr_method))

    # return (np.mean([i[0] for i in all_corr_grades]),
    # (np.mean([i[1][0] for i in all_corr_grades]), np.mean([i[1][1] for i in all_corr_grades])),
    # (np.mean([i[2][0] for i in all_corr_grades]), np.mean([i[2][1] for i in all_corr_grades])))

    # for clip in grouper(Y_predicted_arr, 18):
    #     print(flatten_list(clip))
    # print(flatten_list(Y_test_arr))
    # print(flatten_list(Y_predicted_arr))
    # quit()

    return calculate_corr(flatten_list(Y_test_arr), flatten_list(Y_predicted_arr), method=corr_method)


def create_learning_data_features_and_objective_for_all_subjects(features_df, objective_df):
    # create df for implicit media tagging;
    #               return: X=26*18 (each is FE in terms of features), Y=dido (v_objective for each clip)
    #               X is array of arrays ([[f1...fn], [f1...fn]]), Y is 2-D array of tags: rows*4 (v,a,l,r)

    modified_df = features_df.copy()    # add the objective rating for each row (for each FE)

    # attach 4 new columns to the features_df: v, a, l, r (objective)
    for s in dictionaries.SUBJECTS_IDS:
        for idx, row in objective_df.iterrows():
            modified_df.set_value((s, idx), 'valence_obj', row.valence)
            modified_df.set_value((s, idx), 'arousal_obj', row.arousal)
            modified_df.set_value((s, idx), 'likeability_obj', row.likeability)
            modified_df.set_value((s, idx), 'rewatch_obj', row.rewatch)

    # create X,Y as array of arrays
    mat = modified_df.as_matrix()
    X = []
    Y = []
    for row in mat:
        X.append(row[:-4])
        Y.append(row[-4:])

    return np.array(X), np.array(Y)


def create_learning_data_features_and_objective_for_single_subject(subj_features_df, y_df, axis_name, obj_or_subj='obj'):
    # create df for implicit media tagging;
    #               return: X=18 (each is FE in terms of features), Y=dido (v_objective for each clip)
    #               X is array of arrays ([[f1...fn], [f1...fn]]), Y is 2-D array of tags: rows*4 (v,a,l,r)
    # Y include also the clip_id and subsegment_id (last column)

    modified_df = subj_features_df.copy()    # add the objective rating for each row (for each FE)
    org_index = modified_df.index            # temporarily set index to be just clip_id, for adding v,a,l,r

    if obj_or_subj != 'obj':                 # as opposed to using objective ratings, here we need to keep the multiindex of (subj_id, clip_id)...
        modified_df['subj_id'] = modified_df.index.get_level_values('subj_id')

    modified_df.set_index(modified_df.index.get_level_values('clip_id'), inplace=True)
    modified_df.index = modified_df.index.map(lambda x: int(str(x).split('_')[0]))
    modified_df.index.name = 'clip_id'

    if obj_or_subj != 'obj':                 # ...
        modified_df.reset_index(level=['clip_id'], inplace=True)
        modified_df.set_index(['subj_id', 'clip_id'], inplace=True)

    # attach 4 new columns to the features_df: v, a, l, r (objective)
    for idx, row in y_df.iterrows():
        modified_df.set_value(idx, axis_name+'_obj', row.loc[axis_name])
        # modified_df.set_value(idx, 'valence_obj', row.valence)
        # modified_df.set_value(idx, 'arousal_obj', row.arousal)
        # modified_df.set_value(idx, 'likeability_obj', row.likeability)
        # modified_df.set_value(idx, 'rewatch_obj', row.rewatch)

    modified_df.set_index(org_index, inplace=True)

    # create X,Y as array of arrays
    mat = modified_df.as_matrix()
    X = []
    Y = []
    for row in mat:
        X.append(row[:-1])
        Y.append(row[-1:])

    return np.array(X), np.array(Y)


def split_data(size, cv_model_name):

    # http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut
    if cv_model_name == 'loo':
        from sklearn.cross_validation import LeaveOneOut
        cv_model = LeaveOneOut(size)

    # http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
    if cv_model_name == 'kfold':
        from sklearn.cross_validation import KFold
        cv_model = KFold(size, 10)

    # http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeavePOut.html#sklearn.cross_validation.LeavePOut
    if cv_model_name == 'lpo':
        from sklearn.cross_validation import LeavePOut
        cv_model = LeavePOut(size, 2)

    # The folowwing 3 cross validation models
    # when learning a model *for each subj* leave a clip out (all it's sub-segments), 18
    if cv_model_name == 'LeaveOneClipOutForEachSubject':
        # This one also works for MODEL FOR EACH SUBJ (leaving a whole clip out)
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, NUM_CLIPS)

        for clip in range(NUM_CLIPS):
            test = full_test[clip]
            train = np.setdiff1d(full_arr, test)
            cv_model.append((train, test))

    # EASIEST: when learning a model *over all subj* leave a clip out (all it's sub-segments), 468
    if cv_model_name == 'LeaveOneClipOutForAllSubject':
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, NUM_CLIPS*len(dictionaries.SUBJECTS_IDS))

        for clip in range(NUM_CLIPS*len(dictionaries.SUBJECTS_IDS)):
            test = full_test[clip]
            train = np.setdiff1d(full_arr, test)
            cv_model.append((train, test))

    # when learning a model *over all subj* leave one subj out, 26
    # assuming number of clips is 18
    if cv_model_name == 'LeaveOneSubjOut':
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, len(dictionaries.SUBJECTS_IDS))

        for subj in range(len(dictionaries.SUBJECTS_IDS)):
            test = full_test[subj]
            train = np.setdiff1d(full_arr, test)
            cv_model.append((train, test))

    return cv_model



def run_learning(X_train, Y_train, learning_model, is_normalized=False, args=[]):

    # todo try playing with parameters like alpha

    # http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
    if learning_model == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression(normalize=is_normalized)
        clf.fit(X_train, Y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    elif learning_model.strip() == 'ridge':
        from sklearn.linear_model import Ridge
        clf = Ridge(normalize=is_normalized)
        clf.fit(X_train, Y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    elif learning_model.strip() == 'SVC':
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X_train, Y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    elif learning_model == 'lasso':
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=0.1)
        clf.fit(X_train, Y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    elif learning_model.strip() == 'SVR':
        from sklearn.svm import SVR
        clf = SVR(kernel='rbf')
        clf.fit(X_train, Y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR
    elif learning_model == 'NuSVR':
        from sklearn.svm import NuSVR
        clf = NuSVR(kernel='sigmoid')
        clf.fit(X_train, Y_train)

    return clf


# --------------------------
# ---     Evaluate       ---
# --------------------------

def calculate_corr(actual_y, predicted_y, method):

    if method == 'acc':     # r^2 is f1, corr is acc, p-value is bacc, MCC is 4th
        return f1_score(actual_y, predicted_y), (accuracy_score(actual_y, predicted_y), balanced_accuracy_score(actual_y, predicted_y)), (matthews_corrcoef(actual_y, predicted_y), 0)
    else:
        if np.var(actual_y) == 0:
            return r2_score(actual_y, predicted_y), (0, pearsonr(actual_y, predicted_y)[1]), (se_of_regression(predicted_y, actual_y), 0) # 5 numbers
        return r2_score(actual_y, predicted_y), pearsonr(actual_y, predicted_y), spearmanr(actual_y, predicted_y) # 5 numbers
        # return r2_score(actual_y, predicted_y), pearsonr(actual_y, predicted_y), (se_of_regression(actual_y, predicted_y), 0) # 5 numbers

# --------------------------
# ---      Utils         ---
# --------------------------
def drop_clips(df, clips_list):
    # drops the clips in clips_list from the df. 'isinstance' is for y_df where clip_ids are ints (10) and not strings (10_i)
    return df.reset_index('clip_id')[~df.reset_index('clip_id').clip_id.apply(lambda x: int(x.split('_')[0]) if isinstance(x, str) else x).isin(clips_list)].set_index('clip_id', append=True)

def drop_subjects(df, subjects_list):
    # drops the subjects is sucbjects_list from the df.
    # if 'subj_id' in df.index.names:     # when running over objective_df there's no 'subj_id' column
    return df.reset_index('subj_id')[~df.reset_index('subj_id').subj_id.isin(subjects_list)].set_index('subj_id', append=True).reorder_levels(['subj_id','clip_id'])
    # else:
    #     return df

# --------------------------
# - Implicit Media Tagging -
# --------------------------

def second_learner(df, cv_model_name, learning_model_name):
    # df is results_df
    return_df = df.copy()

    # add column of first learner's x - mean and variance
    df['predicted_mean'] = df.predicted_full.apply(np.mean)
    df['predicted_var'] = df.predicted_full.apply(np.var)
    df['first_learner_x'] = df[['predicted_mean', 'predicted_var']].values.tolist()
    # df.drop(['predicted_mean', 'predicted_var'], axis=1, inplace=True)

    # Cross-Validation
    if cv_model_name == 'LeaveClipOut':
        lpl = LeavePLabelOut(df.index.get_level_values('org_clip'), p=1)
    elif cv_model_name == 'LeaveSubjOut':
        lpl = LeavePLabelOut(df.index.get_level_values('subj_id'), p=1)
    elif cv_model_name == 'LeaveOneClipOfSubj':
        lpl = LeavePLabelOut([str(i + '_' + str(j)) for i, j in df.index.tolist()], p=1)

    for train_index, test_index in lpl:  # leave p clips out
        # Train
        # train_labels = [i for i in set(feat_df.iloc[train_index,:].index.get_level_values('org_clip'))]
        y_train = df.ix[train_index, 'actual_y']
        clf = run_learning(df.ix[train_index, ['predicted_mean', 'predicted_var']], pd.to_numeric(y_train), learning_model_name, is_normalized=True)
        # Test
        test_labels = list_to_unique_list_preserve_order([(i, j) for i, j in df.ix[test_index, :].index.tolist()])
        predicted_y = clf.predict(df.ix[test_index, ['predicted_mean', 'predicted_var']])
        actual_y = df.ix[test_index, 'actual_y'].tolist()
        if cv_model_name in {'LeaveSubjOut', 'LeaveClipOut', 'LeaveOneClipOfSubj'}:
            for idx, (subj_id, org_clip) in enumerate(test_labels):
                # possible to scale predicted_y here
                return_df.loc[(subj_id, org_clip), ['predicted_y', 'actual_y']] = pd.Series([predicted_y[idx], actual_y[idx]]).values

    return return_df


def implicit_media_tagging(df_moments, df_quantized, df_dynamic, df_misc, y_df, model_for_each_subject, clip_drop_list, subj_drop_list,
                           fs_model_name, fs_n_components, pca_each_axis, axis, learning_model_name, cv_model_name, is_second_learner):

    if clip_drop_list:      # drop clips
        [y_df, df_moments, df_quantized, df_dynamic, df_misc] = [drop_clips(df, clip_drop_list) for df in
                                                                 [y_df, df_moments, df_quantized, df_dynamic, df_misc]]
    if subj_drop_list:      # drop subjects
        [y_df, df_moments, df_quantized, df_dynamic, df_misc] = [drop_subjects(df, subj_drop_list) for df in
                                                                 [y_df, df_moments, df_quantized, df_dynamic, df_misc]]
    results_df = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], labels=[[], []], names=['subj_id', 'org_clip']),
                              columns=['predicted_y', 'actual_y', 'predicted_full', 'actual_full'])

    if model_for_each_subject:
        for subj_id in set(y_df.index.get_level_values(level='subj_id')):
            [cur_y_df, cur_df_moments, cur_df_quantized, cur_df_dynamic, cur_df_misc] = [df.loc[[subj_id]] for df in [y_df, df_moments, df_quantized, df_dynamic, df_misc]]

            # PCA
            feat_df = pca(pca_each_axis=pca_each_axis, df_to_pca=[cur_df_moments, cur_df_quantized, cur_df_dynamic],
                          df_not_to_pca=[cur_df_misc], fs_n_components=fs_n_components) if fs_model_name == 'pca' \
                else pd.concat([cur_df_misc, cur_df_moments, cur_df_quantized, cur_df_dynamic], axis=1)

            # Add Y and Original Clip
            feat_df = add_y(add_original_clip(feat_df), cur_y_df, axis)

            # Cross-Validation
            lpl = LeavePLabelOut(feat_df.index.get_level_values('org_clip'), p=1)
            for train_index, test_index in lpl:     # leave p clips out
                # Train
                # train_labels = [i for i in set(feat_df.iloc[train_index,:].index.get_level_values('org_clip'))]
                y_train = feat_df.iloc[train_index, -1]
                if np.var(y_train) == 0:
                    print('subject %s was skipped for 0 variance' % subj_id)
                    continue
                clf = run_learning(feat_df.iloc[train_index,:-1], y_train, learning_model_name, is_normalized=True)
                # Test
                test_label = feat_df.iloc[test_index,:].index.get_level_values('org_clip').values[0]
                predicted_y = clf.predict(feat_df.iloc[test_index,:-1])     # number of segments
                actual_y = feat_df.iloc[test_index, -1].tolist()
                results_df.loc[(subj_id, test_label), ['predicted_y', 'actual_y', 'predicted_full', 'actual_full']] \
                    = pd.Series([np.median(predicted_y), actual_y[0], predicted_y, actual_y]).values

    else:       # NOT a model for each subject
        # PCA
        feat_df = pca(pca_each_axis=pca_each_axis, df_to_pca=[df_moments, df_quantized, df_dynamic], df_not_to_pca=[df_misc],
                      fs_n_components=fs_n_components) if fs_model_name == 'pca' \
            else pd.concat([df_moments, df_quantized, df_dynamic, df_misc], axis=1)

        # Add Y and Original Clip
        feat_df = add_y(add_original_clip(feat_df), y_df, axis)
        seg_size = max([int(s.split('_')[1]) for s in feat_df.index.get_level_values(level='clip_id').tolist()])

        # Cross-Validation
        if cv_model_name == 'LeaveSubjOut':
            lpl = LeavePLabelOut(feat_df.index.get_level_values('subj_id'), p=1)
        elif cv_model_name == 'LeaveClipOut':
            lpl = LeavePLabelOut(feat_df.index.get_level_values('org_clip'), p=1)
        elif cv_model_name == 'LeaveOneClipOfSubj':
            lpl = LeavePLabelOut([str(i+'_'+str(j)) for i,j,k in feat_df.index.tolist()], p=1)

        for train_index, test_index in lpl:  # leave p clips out
            # Train
            # train_labels = [i for i in set(feat_df.iloc[train_index,:].index.get_level_values('org_clip'))]
            y_train = feat_df.iloc[train_index, -1]
            clf = run_learning(feat_df.iloc[train_index, :-1], y_train, learning_model_name, is_normalized=True)
            # Test
            test_labels = list_to_unique_list_preserve_order([(i,j) for i,j,k in feat_df.iloc[test_index, :].index.tolist()])
            predicted_y = clf.predict(feat_df.iloc[test_index, :-1])  # number of segments
            actual_y = feat_df.iloc[test_index, -1].tolist()
            if cv_model_name in {'LeaveSubjOut', 'LeaveClipOut', 'LeaveOneClipOfSubj'}:
                for idx, (subj_id, org_clip) in enumerate(test_labels):
                    # possible to scale predicted_y here
                    results_df.loc[(subj_id, org_clip), ['predicted_y', 'actual_y', 'predicted_full', 'actual_full']] \
                        = pd.Series([np.median(predicted_y[idx:idx + seg_size]), actual_y[idx * seg_size], predicted_y[idx:idx+seg_size], actual_y[idx:idx+seg_size]]).values

    # -- at this point done creating results_df --

    # second learner
    results_df = second_learner(results_df, cv_model_name, learning_model_name) if is_second_learner else results_df

    return results_df

# --------------------------
# ---       Main         ---
# --------------------------

if __name__ == '__main__':

    # load all dfs
    all_features_df, df_moments, df_quantized, df_dynamic, objective_df, ratings_df, big5_df, raw_df = \
        learning_load_all_dfs(use_hl=True, slice_for_specific_blendshapes=False, blendshapse=GOOD_BLENDSHAPES,
                              use_both_for_obj=False)
    print('a')