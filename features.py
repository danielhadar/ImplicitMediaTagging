from preprocessing import *
from utils import *
from scipy import stats as sp

# --------------------------
# ---     Loadings       ---
# --------------------------

def load_all_dfs(org=False):
    ratings_df = load_pickle_to_df(PICKLES_FOLDER + '/ratings.pickle')
    big5_df = load_pickle_to_df(PICKLES_FOLDER + '/big5.pickle')
    objective_df = load_pickle_to_df(PICKLES_FOLDER + '/objective.pickle')

    if org:
        raw_df = load_pickle_to_df(PICKLES_FOLDER + '/org_raw.pickle')
        hl_df = load_pickle_to_df(PICKLES_FOLDER + '/org_raw_with_hl.pickle')
    else:
        raw_df = load_pickle_to_df(PICKLES_FOLDER + '/raw.pickle')
        hl_df = load_pickle_to_df(PICKLES_FOLDER + '/raw_with_hl.pickle')

    return ratings_df, big5_df, objective_df, raw_df, hl_df


# --------------------------
# --  Features: Moments   --
# --------------------------

def create_moments_df_from_raw_df(raw_df, create_over_segmentized):
    pd.options.mode.chained_assignment = None  # default='warn'
    #                    EyeBlink_L_mean  EyeBlink_R_mean  EyeSquint_L_mean  ... var ... skew ... kurt [468r x 204c]
    # subj_id   clip_id
    # 200398733 1               0.465710         0.537161          0.007213

    # return df of all 4 moments: mean, var, skew, kurt
    list_of_all_dfs_to_concat = []

    if create_over_segmentized:
        mean_df = raw_df.iloc[:,2:].groupby(level=[0,1]).mean()
        var_df = raw_df.iloc[:,2:].groupby(level=[0,1]).var()
        skew_df = raw_df.iloc[:,2:].groupby(level=[0,1]).aggregate(sp.skew, bias=False)
        kurt_df = raw_df.iloc[:,2:].groupby(level=[0,1]).aggregate(sp.kurtosis, bias=False)

    else:
        # add original clip id to all sub_segments (i.e. 1 -> 1_1, 1_2, 1_3...)
        raw_df['org_clip_id'] = raw_df.index.get_level_values('clip_id')
        raw_df['org_clip_id'] = raw_df['org_clip_id'].apply(lambda x: x.split('_')[0])
        raw_df.set_index('org_clip_id', append=True, inplace=True)

        # make 'clip_id' (the segmentized) to be not-index and compute moments for the unsegmentized df
        mean_df = raw_df.iloc[:,2:].groupby(level=[0,2]).mean()
        var_df = raw_df.iloc[:,2:].groupby(level=[0,2]).var()
        skew_df = raw_df.iloc[:,2:].groupby(level=[0,2]).aggregate(sp.skew, bias=False)
        kurt_df = raw_df.iloc[:,2:].groupby(level=[0,2]).aggregate(sp.kurtosis, bias=False)
        raw_df.reset_index(level=['clip_id'], inplace=True)

        tmp_mean_df = raw_df.copy()
        tmp_var_df = raw_df.copy()
        tmp_skew_df = raw_df.copy()
        tmp_kurt_df = raw_df.copy()

        # in the original (segmentized) df, set the values. the result is that all cells with the same org_clip will
        # have the same value (i.e. 1_1, 1_2, 1_3... EyeBlink_L is the same)
        # the iteration is along mean_df arbitrary, could be over any MOMENT_df
        for subj_id, org_clip_id in mean_df.index.unique():
            tmp_mean_df.loc[subj_id, org_clip_id].iloc[:,3:] = mean_df.loc[subj_id, org_clip_id].values
            tmp_var_df.loc[subj_id, org_clip_id].iloc[:,3:] = var_df.loc[subj_id, org_clip_id].values
            tmp_skew_df.loc[subj_id, org_clip_id].iloc[:,3:] = skew_df.loc[subj_id, org_clip_id].values
            tmp_kurt_df.loc[subj_id, org_clip_id].iloc[:,3:] = kurt_df.loc[subj_id, org_clip_id].values

        # remove lines where clip_id is '9' and not '9_?'. The reason is that when calculating moments over the
        # entire watching time, the un-sub-segmented clip remains.
        tmp_mean_df = tmp_mean_df[tmp_mean_df.clip_id.str.contains('_')]
        tmp_var_df = tmp_var_df[tmp_var_df.clip_id.str.contains('_')]
        tmp_skew_df = tmp_skew_df[tmp_skew_df.clip_id.str.contains('_')]
        tmp_kurt_df = tmp_kurt_df[tmp_kurt_df.clip_id.str.contains('_')]

        # change clip_id to be index again and discard org_clip_id
        tmp_mean_df.reset_index(level=['org_clip_id'], inplace=True)
        tmp_mean_df.drop('org_clip_id', axis=1, inplace=True)
        tmp_mean_df.set_index('clip_id', append=True, inplace=True)
        tmp_var_df.reset_index(level=['org_clip_id'], inplace=True)
        tmp_var_df.drop('org_clip_id', axis=1, inplace=True)
        tmp_var_df.set_index('clip_id', append=True, inplace=True)
        tmp_skew_df.reset_index(level=['org_clip_id'], inplace=True)
        tmp_skew_df.drop('org_clip_id', axis=1, inplace=True)
        tmp_skew_df.set_index('clip_id', append=True, inplace=True)
        tmp_kurt_df.reset_index(level=['org_clip_id'], inplace=True)
        tmp_kurt_df.drop('org_clip_id', axis=1, inplace=True)
        tmp_kurt_df.set_index('clip_id', append=True, inplace=True)

        mean_df = tmp_mean_df.groupby(level=[0,1]).first().iloc[:,2:]
        var_df = tmp_var_df.groupby(level=[0,1]).first().iloc[:,2:]
        skew_df = tmp_skew_df.groupby(level=[0,1]).first().iloc[:,2:]
        kurt_df = tmp_kurt_df.groupby(level=[0,1]).first().iloc[:,2:]



    # rename columns and stitch together
    mean_df.columns = [name + '_mean' for name in mean_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(mean_df)

    var_df.columns = [name + '_var' for name in var_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(var_df)

    skew_df.columns = [name + '_skew' for name in skew_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(skew_df)

    kurt_df.columns = [name + '_kurt' for name in kurt_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(kurt_df)

    return pd.concat([d for d in list_of_all_dfs_to_concat], axis=1)


# --------------------------
# -  Features: Quantized   -
# --------------------------

def quantize_data_from_raw_df(raw_df, n_quants):
    # for each column (i.e. AU) quantize values using kmeans to n_quants values (0 - n-1).

    quantized_df = raw_df.copy().iloc[:,2:]
    for col in quantized_df:
        # skipping 'time' and 'is_locked', quantize each column (au) in relation to itself
        # the quantization labels ([0] has the centroids)
        quantized_df[col] = cluster.vq.kmeans2(quantized_df[col].values, k=n_quants, iter=300, thresh=1e-02, missing='warn')[1]

    return quantized_df


def create_quantized_features(quantized_df, n_quants):
    # the features are computed for each (subj,clip), and they are:
    #   (1) expression_ratio: amount of expression / total amount of frames     (len = aus)
    #   (2) expression_level: average of each au                                (len = aus)
    #   (3) expression_length:  todo
    #   (4!) expression_average_volume: average of all aus across expression    (len = 1)

    list_of_all_dfs_to_concat = []

    # each (subj,clip) amount of frames for each au
    expression_sum = quantized_df.groupby(level=[0,1]).count()

    # each (subj,clip) amount of non_zeros for each au
    expression_non_zero = quantized_df.groupby(level=[0,1]).apply(lambda col: (col!=0).sum())

    # (1) expression_ratio
    expression_ratio = (expression_non_zero / expression_sum).apply(scale)
    expression_ratio.columns = [name + '_ratio' for name in quantized_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(expression_ratio)

    # (2) expression_level
    expression_level = quantized_df.groupby(level=[0,1]).mean().apply(scale)
    expression_level.columns = [name + '_level' for name in quantized_df.columns.tolist()]
    list_of_all_dfs_to_concat.append(expression_level)

    # (3) expression_average_length
    # todo

    # (4) expression_average_volume
    expression_average_volume = pd.Series(expression_non_zero.mean(axis=1), name='average_volume')
    list_of_all_dfs_to_concat.append(expression_average_volume)

    return pd.concat([d for d in list_of_all_dfs_to_concat], axis=1)


# --------------------------
# --  Features: Dynamic   --
# --------------------------

def create_transition_matrix_dict(quantized_df, n_quants):

    # transition matrix of a single AU, of a single clip, for a single subject looks something like:
    #     |   0  |   1  |   2  |
    #   0 | 0->0 | 0->1 | 0->2 |
    #   1 |      | 1->1 | 1->2 |
    #   2 |      |      | 2->2 |
    # where y-axis is 'from' and x-axis is 'to'
    #
    # this method returns a dict [keys=(subj, clip)] where each value is a dict [keys=(au_1,...,au_n] where
    #       each value is a transition matrix

    from collections import Counter
    print(" ... Computing Transition Matrix ...")

    all_subjects_transition_matrices_dict = {}

    for subj_id, clip_id in quantized_df.index.unique():

        # print("Processing: " + subj_id + ', ' + str(clip_id))
        all_subjects_transition_matrices_dict[(subj_id, clip_id)] = {}

        for au in quantized_df:
            cur_au = quantized_df.loc[(subj_id,clip_id)][au]    # current column
            all_subjects_transition_matrices_dict[(subj_id, clip_id)][au] = np.zeros([n_quants, n_quants], dtype=np.int)

            # for each transition in current au (e.g. x=1, y=2, c=amount of 1->2)
            for (x,y), c in Counter(zip(cur_au, cur_au[1:])).items():
                all_subjects_transition_matrices_dict[(subj_id, clip_id)][au][x, y] = c

    return all_subjects_transition_matrices_dict


def create_dynamic_features(transition_matrix_dict, quantized_features_df):
    # the second argument is used just for it's DF shape...
    # the features are computed for each (subj,clip) for each au are:
    #   (1) change ratio        :   proportion of changing frames
    #   (2) slow change ratio   :   proportion of slow changing frames (e.g. 1->2)
    #   (3) fast change ratio   :   proportion of fast changing frames (e.g. 1->3)

    q_df = quantized_features_df.iloc[:,:0]    # build DF of the shape:  (subj_id, clip_id) x [au_1 ... au_n]

    change_ratio_df = q_df.copy()
    slow_change_ratio_df = q_df.copy()
    fast_change_ratio_df = q_df.copy()

    for (subj_id, clip_id), transition_matrix in transition_matrix_dict.items():
        # print("Processing: " + subj_id + ', ' + str(clip_id))

        for au, au_transition_matrix in transition_matrix.items():
            # main diagonal (e.g. 1->1) are steady transitions
            # secondary diagonal (e.g. 1->2) are slow transitions
            # the rest (e.g. 1->3) are fast transition
            sum_of_frames = sum(sum(au_transition_matrix))

            steady_transitions = sum(np.diagonal(au_transition_matrix))
            slow_transitions = sum(
                np.diagonal(au_transition_matrix, offset=1) + np.diagonal(au_transition_matrix, offset=-1))
            fast_transitions = sum_of_frames - (steady_transitions + slow_transitions)

            change_ratio_df.loc[(subj_id, clip_id), au] = (sum_of_frames - steady_transitions) / sum_of_frames
            slow_change_ratio_df.loc[(subj_id, clip_id), au] = slow_transitions / sum_of_frames
            fast_change_ratio_df.loc[(subj_id, clip_id), au] = fast_transitions / sum_of_frames

    # names_list is because the features dfs columns started off EMPTY and were built based upon the values in the
    #                                                                                               transition matrix
    names_list = change_ratio_df.columns.tolist()

    change_ratio_df.columns = [name + '_change_ratio' for name in names_list]
    slow_change_ratio_df.columns = [name + '_slow_change_ratio' for name in names_list]
    fast_change_ratio_df.columns = [name + '_fast_change_ratio' for name in names_list]

    return pd.concat([change_ratio_df, slow_change_ratio_df, fast_change_ratio_df], axis=1)


# --------------------------
# --   Features: Misc.    --
# --------------------------

def count_blinks(df):
    # returns the number of blinks (max of both eyes
    work_on_df = df.loc[:,'EyeBlink_L':'EyeBlink_R']

    work_on_df['org_clip_id'] = work_on_df.index.get_level_values('clip_id')
    work_on_df['org_clip_id'] = work_on_df['org_clip_id'].apply(lambda x: x.split('_')[0])
    work_on_df.set_index('org_clip_id', append=True, inplace=True)
    work_on_df.reset_index(level=['clip_id'], inplace=True)

    # work_on_df = work_on_df.copy().drop_duplicates(subset='clip_id').drop(['EyeBlink_L','EyeBlink_R'],axis=1)
    return_df = work_on_df.copy().reset_index().drop_duplicates(subset=['subj_id','clip_id']).drop(['EyeBlink_L','EyeBlink_R'],axis=1).set_index(['subj_id','org_clip_id'])

    for idx, data in work_on_df.groupby(level=[0,1]):
        return_df.loc[idx, "blinks"] = max(count_peaks(data.loc[:,'EyeBlink_L'].tolist()),
                                           count_peaks(data.loc[:,'EyeBlink_R'].tolist()))

    return_df.reset_index(level=['org_clip_id'], inplace=True)
    return_df.drop(['org_clip_id'], axis=1, inplace=True)
    return_df.set_index('clip_id', append=True, inplace=True)

    return return_df


def count_smiles(df, th=0.75):
    # returns the number of smiles (number of peaks above th)
    work_on_df = df.loc[:,'MouthSmile_L':'MouthSmile_R']

    work_on_df['org_clip_id'] = work_on_df.index.get_level_values('clip_id')
    work_on_df['org_clip_id'] = work_on_df['org_clip_id'].apply(lambda x: x.split('_')[0])
    work_on_df.set_index('org_clip_id', append=True, inplace=True)
    work_on_df.reset_index(level=['clip_id'], inplace=True)

    # work_on_df = work_on_df.copy().drop_duplicates(subset='clip_id').drop(['EyeBlink_L','EyeBlink_R'],axis=1)
    return_df = work_on_df.copy().reset_index().drop_duplicates(subset=['subj_id','clip_id']).drop(['MouthSmile_L','MouthSmile_R'],axis=1).set_index(['subj_id','org_clip_id'])

    for idx, data in work_on_df.groupby(level=[0,1]):
        return_df.loc[idx, "smiles"] = max(count_peaks(data.loc[:,'MouthSmile_L'].tolist(), delta=th),
                                           count_peaks(data.loc[:,'MouthSmile_R'].tolist(), delta=th))

    return_df.reset_index(level=['org_clip_id'], inplace=True)
    return_df.drop(['org_clip_id'], axis=1, inplace=True)
    return_df.set_index('clip_id', append=True, inplace=True)

    return return_df


# --------------------------
# ---        Main        ---
# --------------------------

def create_features(use_hl=True, slice_for_specific_bs=False, bs_list=[],
                    create_moments_over_hl=False, create_moments_over_segmentized=False, use_overlap=False):

    ratings_df, big5_df, objective_df, raw_df, hl_df = load_all_dfs(org=False)
    ol_df = load_pickle_to_df(PICKLES_FOLDER + '/overlap_df.pickle')

    if slice_for_specific_bs:
        hl_df = slice_features_df_for_specific_blendshapes(hl_df, bs_list)
        raw_df = slice_features_df_for_specific_blendshapes(raw_df, bs_list)

    if use_hl:
        work_df = hl_df
        work_df_no_slice = hl_df.copy()
        work_on = 'hl'
    else:
        work_df = raw_df
        raw_df_no_slice = raw_df.copy()
        work_on = 'watch'

    if use_overlap:
        work_df = ol_df
        work_df_no_slice = ol_df.copy()

    if use_hl:
        # -- moments features --
        print(" -- Moments -- ")
        if create_moments_over_hl:
            # either moments are calculated over the 'hl' part only or over the entire watching time ('watch'+'hl')
            df = create_moments_df_from_raw_df(work_df.xs('hl', level=2), create_over_segmentized=create_moments_over_segmentized)
        else:
            df = create_moments_df_from_raw_df(
                work_df[work_df.index.get_level_values('response_type').isin(['watch', 'hl'])].reset_index(level=2, drop=True),
                create_over_segmentized=create_moments_over_segmentized)
        export_df_to_pickle(df, PICKLES_FOLDER + '/features/moments_features_hl.pickle')

        # -- quantized the data --
        print(" -- Quantized --")
        df = quantize_data_from_raw_df(work_df.xs('hl', level=2), 4)
        export_df_to_pickle(df, PICKLES_FOLDER + '/4-quantized_data_hl.pickle')

        # -- quantized features --
        quantized_watch_df = load_pickle_to_df(PICKLES_FOLDER + '/4-quantized_data_hl.pickle')
        df = create_quantized_features(quantized_watch_df, 4)
        export_df_to_pickle(df, PICKLES_FOLDER + '/features/quantized_features_hl.pickle')

        # -- dynamic features (based on quantized data) --
        print(" -- Dynamic --")
        quantized_watch_df = load_pickle_to_df(PICKLES_FOLDER + '/4-quantized_data_hl.pickle')
        transition_matrix_dict = create_transition_matrix_dict(quantized_watch_df, 4)
        export_dict_to_pickle(transition_matrix_dict, PICKLES_FOLDER + '/transition_matrix_dict_hl.pickle')

        quantized_features_df = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features_hl.pickle')
        transition_matrix_dict = load_pickle_to_dict(PICKLES_FOLDER + '/transition_matrix_dict_hl.pickle')
        dynamic_features_df = create_dynamic_features(transition_matrix_dict, quantized_features_df)
        export_df_to_pickle(dynamic_features_df, PICKLES_FOLDER + '/features/dynamic_features_hl.pickle')

        # -- miscellaneous features --
        print(" -- Miscellaneous --")
        blink_df = count_blinks(work_df_no_slice.xs('hl', level=2))
        smile_df = count_smiles(work_df_no_slice.xs('hl', level=2))
        misc_df = pd.concat([blink_df, smile_df], axis=1)
        export_df_to_pickle(misc_df, PICKLES_FOLDER + '/features/misc_features_hl.pickle')

        # -- create all features df --
        print(" > Creating All Features DF")
        df_moments = load_pickle_to_df(PICKLES_FOLDER + '/features/moments_features_hl.pickle')
        df_quantized = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features_hl.pickle')
        df_dynamic = load_pickle_to_df(PICKLES_FOLDER + '/features/dynamic_features_hl.pickle')
        df_misc = load_pickle_to_df(PICKLES_FOLDER + '/features/misc_features_hl.pickle')
        all_features_df = pd.concat([df_moments, df_quantized, df_dynamic, df_misc], axis=1)
        all_features_df.to_csv(CSV_FOLDER + '/all_features_df_hl.csv')
        export_df_to_pickle(all_features_df, PICKLES_FOLDER + '/features/all_features_hl.pickle')

    else:       # not using hl
        # -- moments features --
        print(" > Starting Moments Features")
        df = create_moments_df_from_raw_df(raw_df.xs('watch', level=2), create_over_segmentized=create_moments_over_segmentized)
        export_df_to_pickle(df, PICKLES_FOLDER + '/features/moments_features.pickle')

        # -- quantized the data --
        print(" > Starting Quantized Features")
        df = quantize_data_from_raw_df(raw_df.xs('watch', level=2), 4)
        export_df_to_pickle(df, PICKLES_FOLDER + '/4-quantized_data.pickle')

        # -- quantized features --
        quantized_watch_df = load_pickle_to_df(PICKLES_FOLDER + '/4-quantized_data.pickle')
        df = create_quantized_features(quantized_watch_df, 4)
        export_df_to_pickle(df, PICKLES_FOLDER + '/features/quantized_features.pickle')

        # -- dynamic features (based on quantized data) --
        print(" > Starting Dynamic Features")
        quantized_watch_df = load_pickle_to_df(PICKLES_FOLDER + '/4-quantized_data.pickle')
        transition_matrix_dict = create_transition_matrix_dict(quantized_watch_df, 4)
        export_dict_to_pickle(transition_matrix_dict, PICKLES_FOLDER + '/transition_matrix_dict.pickle')

        quantized_features_df = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features.pickle')
        transition_matrix_dict = load_pickle_to_dict(PICKLES_FOLDER + '/transition_matrix_dict.pickle')
        dynamic_features_df = create_dynamic_features(transition_matrix_dict, quantized_features_df)
        export_df_to_pickle(dynamic_features_df, PICKLES_FOLDER + '/features/dynamic_features.pickle')

        # -- miscellaneous features --
        print(" -- Miscellaneous --")
        blink_df = count_blinks(raw_df_no_slice.xs('watch', level=2))
        smile_df = count_smiles(raw_df_no_slice.xs('watch', level=2))
        misc_df = pd.concat([blink_df, smile_df], axis=1)
        export_df_to_pickle(misc_df, PICKLES_FOLDER + '/features/misc_features.pickle')

        # -- create all features df --
        print(" > Creating All Features DF")
        df_moments = load_pickle_to_df(PICKLES_FOLDER + '/features/moments_features.pickle')
        df_quantized = load_pickle_to_df(PICKLES_FOLDER + '/features/quantized_features.pickle')
        df_dynamic = load_pickle_to_df(PICKLES_FOLDER + '/features/dynamic_features.pickle')
        df_misc = load_pickle_to_df(PICKLES_FOLDER + '/features/misc_features.pickle')
        all_features_df = pd.concat([df_moments, df_quantized, df_dynamic, df_misc], axis=1)
        all_features_df.to_csv(CSV_FOLDER + '/all_features_df.csv')
        export_df_to_pickle(all_features_df, PICKLES_FOLDER + '/features/all_features.pickle')


if __name__ == '__main__':
    ratings_df, big5_df, objective_df, raw_df, hl_df = load_all_dfs(org=False)
    raw_df_no_slice = raw_df.copy()
    hl_df_no_slice = hl_df.copy()
    blink_df = count_blinks(hl_df_no_slice.xs('hl', level=2))
    smile_df = count_smiles(hl_df_no_slice.xs('hl', level=2))
    print(blink_df)
    print('---------')
    print(smile_df)
    print('---------')
    misc_df = pd.concat([blink_df, smile_df], axis=1)
    print(misc_df)