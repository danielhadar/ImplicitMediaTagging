from features import *
from utils import *



# --------------------------
# ---     Loadings       ---
# --------------------------

def learning_load_all_dfs(use_hl=False, use_both_for_obj=False):
    if use_hl:
        all_features_df = load_pickle_to_df(PICKLES_FOLDER + '/features/all_features_hl.pickle')
        df_moments = load_pickle_to_df(PICKLES_FOLDER + '/features/moments_features_hl.pickle')
        df_quantized = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features_hl.pickle')
        df_dynamic = load_pickle_to_df(PICKLES_FOLDER + '/features/dynamic_features_hl.pickle')
        df_misc = load_pickle_to_df(PICKLES_FOLDER + '/features/misc_features_hl.pickle')

    else:
        all_features_df = load_pickle_to_df(PICKLES_FOLDER + '/features/all_features.pickle')
        df_moments = load_pickle_to_df(PICKLES_FOLDER + '/features/moments_features.pickle')
        df_quantized = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features.pickle')
        df_dynamic = load_pickle_to_df(PICKLES_FOLDER + '/features/dynamic_features.pickle')
        df_misc = load_pickle_to_df(PICKLES_FOLDER + '/features/misc_features.pickle')

    # if slice_for_specific_blendshapes:
    #     all_features_df = slice_features_df_for_specific_blendshapes(all_features_df, blendshapse)
    #     df_moments = slice_features_df_for_specific_blendshapes(df_moments, blendshapse)
    #     df_quantized = slice_features_df_for_specific_blendshapes(df_quantized, blendshapse)
    #     df_dynamic = slice_features_df_for_specific_blendshapes(df_dynamic, blendshapse)

    if use_both_for_obj:
        objective_df = load_pickle_to_df(PICKLES_FOLDER + '/objective_both.pickle')
    else:
        objective_df = load_pickle_to_df(PICKLES_FOLDER + '/objective.pickle')

    majority_objective_df = load_pickle_to_df(PICKLES_FOLDER + '/majority_objective.pickle')
    ratings_df = load_pickle_to_df(PICKLES_FOLDER + '/ratings.pickle')
    big5_df = load_pickle_to_df(PICKLES_FOLDER + '/big5.pickle')
    raw_df = load_pickle_to_df(PICKLES_FOLDER + '/raw.pickle')

    print(" > Done Loading!\n")

    return all_features_df, df_moments, df_quantized, df_dynamic, df_misc, objective_df, ratings_df, big5_df, raw_df, majority_objective_df



# --------------------------
# --  Feature Selection   --
# --------------------------

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

    if is_scaled:
        from sklearn import preprocessing
        X = preprocessing.scale(X)

    Y_predicted_arr = []
    Y_test_arr = []

    # train-set and test-set are identical (and equal to the entire set)
    if cv_model_name == 'none':
        clf = run_learning(X, Y, learning_model_name)
        Y_predicted_arr.append(clf.predict(X))       # add predicted y to predicted array
        Y_test_arr.append(Y)                         # add actual    y to actual    array

    else:
        cv_model_idx_list = split_data(len(Y), cv_model_name)
        for train_idx, test_idx in cv_model_idx_list:

            # split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # learn and append y-predicted AND y-true values
            clf = run_learning(X_train, Y_train, learning_model_name, is_normalized)
            Y_predicted = clf.predict(X_test)

            # calculate segment size
            seg_size = int(len(X)/(NUM_CLIPS*len(SUBJECTS_IDS)))
            if seg_size < 1:    # when computing a model for each subj, len(x) is as long as one subject's data
                seg_size = int(len(X)/NUM_CLIPS)


            if is_second_learner:
                second_learner_X = []
                first_learner_predicted_Y = clf.predict(X_train)              # fit-predict
                chunks_iter = grouper(first_learner_predicted_Y, seg_size)    # iterator of chunks of seg_size
                for clip in chunks_iter:
                    second_learner_X.append(
                        (np.median(clip), np.var(clip)))                        # create X for 2nd learner

                clf2 = run_learning(second_learner_X, Y_train[0::seg_size],
                                    learning_model_name, is_normalized)        # run 2nd learner

                # -- at this point the 2nd model was computed --

                chunks_iter = grouper(Y_predicted, seg_size)                  # iterator of chunks of seg_size
                for clip in chunks_iter:                                      # learn each clip
                    Y_predicted_arr.append(clf2.predict((np.mean(clip), np.var(clip))))

                Y_test_arr.extend(Y_test[0::seg_size])

            else:
                # compute a single grade for each clip (a clip is the size of seg_size)
                if use_single_predicted_Y_foreach_clip:
                    chunks_iter = grouper(Y_predicted, seg_size)    # iterator of chunks of seg_size
                    for clip in chunks_iter:
                        Y_predicted_arr.append(np.median(clip))
                    Y_test_arr.extend(Y_test[0::seg_size])
                    print(SUBJECTS_IDS[int(test_idx[0]/54)])
                    print(flatten_list(Y_test_arr))
                    print(flatten_list(Y_predicted_arr))

                else:
                    Y_predicted_arr.append(Y_predicted)     # add predicted y to predicted array
                    Y_test_arr.append(Y_test)               # add actual    y to actual    array

    return_val = calculate_corr(flatten_list(Y_test_arr), flatten_list(Y_predicted_arr), method=corr_method)
    return return_val
    # return calculate_corr(flatten_list(Y_test_arr), flatten_list(Y_predicted_arr), method=corr_method)


def create_learning_data_features_and_objective_for_all_subjects(features_df, objective_df):
    # create df for implicit media tagging;
    #               return: X=26*18 (each is FE in terms of features), Y=dido (v_objective for each clip)
    #               X is array of arrays ([[f1...fn], [f1...fn]]), Y is 2-D array of tags: rows*4 (v,a,l,r)

    modified_df = features_df.copy()    # add the objective rating for each row (for each FE)

    # attach 4 new columns to the features_df: v, a, l, r (objective)
    for s in SUBJECTS_IDS:
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
    # when learning a model *for each subj* leave a clip out (all it's sub-segments)
    if cv_model_name == 'LeaveOneClipOutForEachSubject':
        # This one also works for MODEL FOR EACH SUBJ (leaving a whole clip out)
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, NUM_CLIPS)

        for clip in range(NUM_CLIPS):
            test = full_test[clip]
            train = np.setdiff1d(full_arr, test)
            cv_model.append((train, test))

    # EASIEST: when learning a model *over all subj* leave a clip out (all it's sub-segments)
    if cv_model_name == 'LeaveOneClipOutForAllSubject':
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, NUM_CLIPS*len(SUBJECTS_IDS))

        for clip in range(NUM_CLIPS*len(SUBJECTS_IDS)):
            test = full_test[clip]
            train = np.setdiff1d(full_arr, test)
            cv_model.append((train, test))

    # when learning a model *over all subj* leave one subj out
    # assuming number of clips is 18
    if cv_model_name == 'LeaveOneSubjOut':
        cv_model = []
        full_arr = np.array([i for i in range(size)])
        full_test = np.array_split(full_arr, len(SUBJECTS_IDS))

        for subj in range(len(SUBJECTS_IDS)):
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
    if method == 'acc':
        return 0, (accuracy_score(actual_y, predicted_y), 0), (0, 0)    # returned as prearson's value
    else:
        return r2_score(actual_y, predicted_y), pearsonr(actual_y, predicted_y), spearmanr(actual_y, predicted_y) # 5 numbers


# --------------------------
# - Implicit Media Tagging -
# --------------------------

def implicit_media_tagging(df_moments, df_quantized, df_dynamic, df_misc, y_df, obj_or_subj,
                           scale_x, model_for_each_subject, to_drop_list,
                           fs_model_name, fs_n_components, axis, learning_model_name, cv_model_name, is_second_learner,
                           f=None, use_single_predicted_Y_foreach_clip=False, corr_method='normal'):

    if model_for_each_subject:

        subjects_corr = []

        for subj in SUBJECTS_IDS:

            clips_drop = to_drop_list
            if obj_or_subj == 'subj':
                cur_y_df = y_df.loc[[subj]]
            else:
                cur_y_df = y_df

            if scale_x:     # scaling WITHIN subject. could be changed to scaled over all subejcts
                X_moments, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_moments.loc[[subj]].drop(clips_drop).apply(scale), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_quantized, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_quantized.loc[[subj]].drop(clips_drop).apply(scale), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_dynamic, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_dynamic.loc[[subj]].drop(clips_drop).apply(scale), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_misc, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_misc.loc[[subj]].drop(clips_drop).apply(scale), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
            else:
                X_moments, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_moments.loc[[subj]].drop(clips_drop), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_quantized, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_quantized.loc[[subj]].drop(clips_drop), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_dynamic, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_dynamic.loc[[subj]].drop(clips_drop), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)
                X_misc, Y = create_learning_data_features_and_objective_for_single_subject \
                    (df_misc.loc[[subj]].drop(clips_drop), cur_y_df.drop(clips_drop), axis[0].strip(), obj_or_subj=obj_or_subj)

            x_arr = []
            for X in [X_moments, X_quantized, X_dynamic]:
                if np.shape(X)[1] > fs_n_components:    # handles case of too short X_features in relation to pca
                    tmp_x = feature_selection(X, fs_model_name, n_components=fs_n_components)
                else:
                    tmp_x = feature_selection(X, fs_model_name, n_components=np.shape(X)[1])

                x_arr.append(tmp_x)
            x_arr.append(X_misc)

            feat = np.concatenate(x_arr, axis=1)    # <<<

            # appended = np.append(np.append(X_moments, X_quantized, axis=1), X_dynamic, axis=1)
            # for i in range(len(appended)):
            #     appended[i] = scale(appended[i])
            # feat = feature_selection(appended, fs_model_name, n_components=fs_n_components)

            if np.var(Y) == 0:
                print('subject %s was skipped for 0 variance' % subj)
                continue

            r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) \
                = run(feat, Y, learning_model_name, cv_model_name,
                      is_scaled=True, is_normalized=True, f=f,
                      use_single_predicted_Y_foreach_clip=use_single_predicted_Y_foreach_clip,
                      is_second_learner=is_second_learner, corr_method=corr_method)

            subjects_corr.append((pearsonr_val, pearsonr_p_val, subj, r2))

        return subjects_corr

    else:
        subj_drop = []
        if subj_drop:
            SUBJECTS_IDS[:-len(subj_drop)]
        X_misc, Y = create_learning_data_features_and_objective_for_single_subject(df_misc.drop(subj_drop), y_df, axis[0].strip(), obj_or_subj=obj_or_subj)
        X_moments, Y = create_learning_data_features_and_objective_for_single_subject(df_moments.drop(subj_drop), y_df, axis[0].strip(), obj_or_subj=obj_or_subj)
        X_quantized, Y = create_learning_data_features_and_objective_for_single_subject(df_quantized.drop(subj_drop), y_df, axis[0].strip(), obj_or_subj=obj_or_subj)
        X_dynamic, Y = create_learning_data_features_and_objective_for_single_subject(df_dynamic.drop(subj_drop), y_df, axis[0].strip(), obj_or_subj=obj_or_subj)

        # (a) feature selection over each type of feature separately
        x_arr = []
        for X in [X_moments, X_quantized, X_dynamic]:
            if np.shape(X)[1] > fs_n_components:
                tmp_x = feature_selection(X, fs_model_name, n_components=fs_n_components)
            else:
                tmp_x = feature_selection(X, fs_model_name, n_components=np.shape(X)[1])
            x_arr.append(tmp_x)
        x_arr.append(X_misc)
        feat = np.concatenate(x_arr, axis=1)    # <<<

        # (b) feature selection over all features
        # feat = np.concatenate([X_misc, X_moments, X_quantized, X_dynamic], axis=1)
        # feat = feature_selection(feat, fs_model_name, n_components=fs_n_components)

        r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) \
            = run(feat, Y, learning_model_name, cv_model_name,
                  is_scaled=True, is_normalized=True, f=f,
                  use_single_predicted_Y_foreach_clip=use_single_predicted_Y_foreach_clip,
                  is_second_learner=is_second_learner, corr_method=corr_method)

        return [[pearsonr_val, pearsonr_p_val, '', r2]]

        # print(learning_model_name + ', ' + cv_model_name
        #       + ', ' + fs_model_name + str(fs_n_components)  + ', ' + axis[0]
        #       + ', %.3f, ' % pearsonr_val + '%.3f' % pearsonr_p_val)


# --------------------------
# ---       Main         ---
# --------------------------

if __name__ == '__main__':

    # load all dfs
    all_features_df, df_moments, df_quantized, df_dynamic, objective_df, ratings_df, big5_df, raw_df = \
        learning_load_all_dfs(use_hl=True, slice_for_specific_blendshapes=False, blendshapse=GOOD_BLENDSHAPES,
                              use_both_for_obj=False)
    print('a')