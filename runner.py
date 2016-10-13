from preprocessing import *
from features import *
from learning import *
from time import time
from datetime import datetime
import sys
import dictionaries
import os


import warnings
warnings.filterwarnings("ignore")

# LEARNING_MODELS = ['linear_regression', 'ridge', 'lasso', 'SVR', 'NuSVR']
# LEARNING_MODELS = ['linear_regression', 'ridge            ', 'SVR              ']
# LEARNING_MODELS = ['linear_regression', 'ridge            ']
# LEARNING_MODELS = ['SVC']
# LEARNING_MODELS = ['SVR']
# LEARNING_MODELS = ['ridge']
LEARNING_MODELS = ['linear_regression']

# CV_MODELS = ['LeaveSubjOut', 'LeaveClipOut', 'LeaveOneClipOfSubj']
# CV_MODELS = ['LeaveSubjOut']
CV_MODELS = ['LeaveClipOut']
# CV_MODELS = ['LeaveOneClipOfSubj']


# RATINGS_AXES = [('valence    ',0), ('arousal    ',1), ('likeability',2), ('rewatch    ',3)]
# RATINGS_AXES = [('valence    ',0)]
RATINGS_AXES = [('arousal    ',1)]
# RATINGS_AXES = [('likeability',2)]
# RATINGS_AXES = [('rewatch',3)]
# RATINGS_AXES = [('valence    ',0), ('arousal    ',1)]
# RATINGS_AXES = [('likeability',2), ('rewatch    ',3)]

# FS_MODELS = ['none', 'pca']
FS_MODELS = ['pca']
# FS_MODELS = ['none']

def mega_runner(run_id, f, run_preprocessing, is_hl_in_preprocessing,
                set_win_size, hl_margins, is_smart_hl, run_segmentize, is_hl, segments_length, run_overlap, overlap_percent,
                run_features, is_hl_in_features, create_moments_over_segmentized, is_slice_for_specific_blendshapes, which_blendshapes,
                use_overlap, run_learning, obj_or_subj, is_hl_in_learning, is_both_for_obj, scale_y,
                is_model_for_each_subject, clip_drop_list, subj_drop_list, fs_models_list, fs_n_components_range, pca_each_axis,
                learning_models_list, ratings_axes_list, cv_models_list, is_second_learner, is_majority_vote, scale_predicted_y_by,
                given_dfs=False, dfs=None, subj_name=None):     # remove subj_name

    if given_dfs:       # dfs were given, go straight to learning
        run_preprocessing = False; set_win_size = False; run_segmentize = False; run_overlap = False; run_features = False

    print(" > Starting Preprocessing...")
    if run_preprocessing:
        create_raw(is_hl=is_hl_in_preprocessing)
        # f.write("run_preprocessing=True, with: is_hl=%s \n" % (is_hl_in_preprocessing))
    # else:
        # f.write("run_preprocessing=False \n")
    print("     Done Preprocessing!\n")


    print(" > Starting Set Window Size... ")
    if set_win_size:
        set_window_size(hl_margins, is_smart_hl)
        # f.write("set_win_size=True, with: hl_margins=(%i, %i) \n" % (hl_margins[0], hl_margins[1]))
    # else:
        # f.write("set_win_size=False \n")
    print("     Done Set Window Size!\n")


    print(" > Starting Segmentizing... ")
    if run_segmentize:
        segmentize(seg_length=segments_length, is_hl=is_hl)
        # f.write("run_segmentize=True, with: seg_len=%i \n" % (segments_length))
    # else:
        # f.write("run_segmentize=False \n")
    print("     Done Segmentizing!\n")


    print(" > Starting Overlap... ")
    if run_overlap:
        add_overlap(overlap_percent=overlap_percent, seg_length=segments_length, is_hl=is_hl)
        # f.write("run_overlap=True, with: ol_percent=%i \n" % (overlap_percent))
    # else:
        # f.write("run_overlap=False \n")
    print("     Done Adding Overlap!\n")


    print(" > Starting Creating Features... ")
    if run_features:
        create_features(use_hl=is_hl_in_features, slice_for_specific_bs=is_slice_for_specific_blendshapes, bs_list=which_blendshapes,
                        create_moments_over_segmentized=create_moments_over_segmentized, use_overlap=use_overlap)
        # f.write("run_features=True, with: is_hl=%s , specific_blendshapes=%s \n" % (is_hl_in_features, is_slice_for_specific_blendshapes))
    # else:
        # f.write("run_features=False \n")
    print("     Done Creating Features!\n")


    print(" L e a r n i n g ... ")
    if run_learning:
        all_features_df, df_moments, df_quantized, df_dynamic, df_misc, objective_df, ratings_df, big5_df, raw_df, majority_df = \
            learning_load_all_dfs(use_hl=is_hl_in_learning, use_both_for_obj=is_both_for_obj)

        if given_dfs:
            df_moments, df_quantized, df_dynamic, df_misc = dfs

        y_df = objective_df if obj_or_subj == 'obj' else ratings_df
        corr_method = 'normal'  # r2, pearson, spearman
        if is_majority_vote:
            y_df = majority_df
            learning_models_list = ['SVC']
            corr_method = 'acc'

        elif scale_y:
            y_df = y_df.apply(scale)

        for cv_model_name in cv_models_list:
            for fs_model_name in fs_models_list:
                for axis in ratings_axes_list:
                    for learning_model_name in learning_models_list:

                        max_result_for_pca = (-inf,-inf,-inf,-inf)
                        max_df = [None, 0]
                        for fs_n_components in fs_n_components_range:

                            results_df = implicit_media_tagging(df_moments, df_quantized, df_dynamic, df_misc, y_df,
                                                                   model_for_each_subject=is_model_for_each_subject,
                                                                   clip_drop_list=clip_drop_list, subj_drop_list=subj_drop_list, fs_model_name=fs_model_name,
                                                                   fs_n_components=fs_n_components, pca_each_axis=pca_each_axis, axis=axis,
                                                                   learning_model_name=learning_model_name, cv_model_name=cv_model_name,
                                                                   is_second_learner=is_second_learner)

                            # Scale predicted_y
                            results_df = scale_column_by(results_df, 'predicted_y', scale_predicted_y_by, is_majority_vote)     # 'org_clip', 'subj_id' or None

                            # Calculate correlations
                            r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) = \
                                calculate_corr(results_df.actual_y.tolist(), results_df.predicted_y.tolist(),
                                               method=corr_method)
                            subjects_corr = [r2, pearsonr_val, pearsonr_p_val, spearman_val]

                            # should usually comment-out - designated for printing subject's score for subj_aside. added it for checking the pca thing.
                            # df_subj_slice = results_df.loc[results_df.index.get_level_values('subj_id') == subj_name]
                            # r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) = calculate_corr(df_subj_slice.actual_y.tolist(), df_subj_slice.predicted_y.tolist(), method='normal')
                            # subjects_corr = [r2, pearsonr_val, pearsonr_p_val, spearman_val]
                            # print("%s, %s%.2i, %s, %s, %s: %.3f, (p=%.3f), r^2=%.3f, spearman=%.3f" %
                            #       (learning_model_name, fs_model_name, fs_n_components, cv_model_name, axis[0], scale_predicted_y_by,
                            #        subjects_corr[1], subjects_corr[2], subjects_corr[0], subjects_corr[3]))
                            # f.write("%.3f, %.3f, %.3f, %.3f, %2i. %s\n" % (subjects_corr[1], subjects_corr[2], subjects_corr[0], subjects_corr[3], fs_n_components, axis[0]))
                            # until here

                            if subjects_corr[1] > max_result_for_pca[1]:
                                max_result_for_pca = subjects_corr
                                max_df = [results_df, fs_n_components]

                        subjects_corr = max_result_for_pca

                        if f:
                            f.write("%.3f, %.3f, %.3f, %.3f, %2i. %s\n" % (subjects_corr[1], subjects_corr[2], subjects_corr[0], subjects_corr[3], fs_n_components, axis[0]))
                        # max_df[0].to_csv(LOG_FOLDER + 'results_df' + run_id + '.csv')

                        if corr_method == 'acc':
                            print("%s, %s%.2i, %s, %s, %s: MCC=%.3f, BACC=%.3f, f1=%.3f, ACC=%.3f" %
                                  (learning_model_name, fs_model_name, max_df[1], cv_model_name, axis[0], scale_predicted_y_by,
                                   subjects_corr[1], subjects_corr[2], subjects_corr[0], subjects_corr[3]))
                        # else:
                        #     print("%s, %s%.2i, %s, %s, %s: %.3f, (p=%.3f), r^2=%.3f, spearman=%.3f" %
                        #           (learning_model_name, fs_model_name, max_df[1], cv_model_name, axis[0], scale_predicted_y_by,
                        #            subjects_corr[1], subjects_corr[2], subjects_corr[0], subjects_corr[3]))

    return subjects_corr, max_df[0], [df_moments, df_quantized, df_dynamic, df_misc], max_df[1]



if __name__ == '__main__':

    # run_type = 'massive'
    # run_type = 'solo'
    run_type = 'cross_validation'

    # for parallel running, run with different argv every time (none-2-3-4)
    run_id = ""
    if not sys.argv[-2].isnumeric():
        dictionaries.PICKLES_FOLDER += "/"
    else:
        run_id = sys.argv[-2]
        dictionaries.PICKLES_FOLDER += str(run_id) + "/"

    if sys.argv[-1] == 'v':
        RATINGS_AXES = [('valence    ',0)]
    elif sys.argv[-1] == 'a':
        RATINGS_AXES = [('arousal    ',1)]
    elif sys.argv[-1] == 'l':
        RATINGS_AXES = [('likeability',2)]
    elif sys.argv[-1] == 'r':
        RATINGS_AXES = [('rewatch    ',3)]


    if run_type == 'massive':
        with open(LOG_FOLDER + 'log_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', 'w') as f:
            f.write("win size, smart hl, seg length, overlap, blendshapes, obj or subj, model for each subject, 2nd learner, majority vote, pca each axis, scale_by, V, , , , A, , , , L, , , , R, , , ,\n")

            # for hl_margins in [(5,1), (3,1)]:
            for hl_margins in [(5,1)]:
            # for hl_margins in [(3,1)]:
            #     for is_smart_hl in [True, False]:
                for is_smart_hl in [True]:
                # for is_smart_hl in [False]:
                    setwinsize = True
                    # for seg_len in [10,30,50]:
                    for seg_len in [10]:
                        runsegmentize = True
                        # for ol_percent in [0, 25, 50, 75]:
                        # for ol_percent in [25, 50, 75]:
                        for ol_percent in [50, 75]:
                        # for ol_percent in [50, 75, 0, 25]:
                            runoverlap = True if ol_percent > 0 else False
                            # for bs in [(BLENDSHAPES, "all"), (MY_BS, "my")]:
                            for bs in [(MY_BS, "my")]:
                                runfeatures = True
                                # for obj_or_subj in ['obj', 'subj']:
                                for obj_or_subj in ['obj']:
                                    # for model_for_each_subj in [True, False]:
                                    for model_for_each_subj in [True]:
                                        # for is_second_learner in [True, False]:
                                        for is_second_learner in [True]:
                                            # for is_majority_vote in [False, True]:
                                            for is_majority_vote in [False]:
                                                for pca_per_axis in [True, False]:
                                                    # for scale_predicted_y_by in ['org_clip', 'subj_id', None]:
                                                    for scale_predicted_y_by in ['org_clip']:

                                                        f.write("%s, %s, %i, %s, %s, %s, %s, %s, %s, %s, %s, " %
                                                                (str(hl_margins).replace(', ','-'), is_smart_hl, seg_len, ol_percent, bs[1],
                                                                 obj_or_subj,
                                                                 model_for_each_subj, is_second_learner, is_majority_vote,
                                                                 pca_per_axis, scale_predicted_y_by))

                                                        mega_runner(str(run_id), f, run_preprocessing=False, is_hl_in_preprocessing=False,

                                                                    set_win_size=setwinsize, hl_margins=hl_margins, is_smart_hl=is_smart_hl,

                                                                    run_segmentize=runsegmentize, is_hl=True, segments_length=seg_len,

                                                                    run_overlap=runoverlap, overlap_percent=ol_percent,

                                                                    run_features=runfeatures, is_hl_in_features=True,
                                                                    create_moments_over_segmentized=False,
                                                                    is_slice_for_specific_blendshapes=True,
                                                                    which_blendshapes=bs[0], use_overlap=True if ol_percent > 0 else False,

                                                                    run_learning=True, obj_or_subj=obj_or_subj,
                                                                    is_hl_in_learning=True,
                                                                    is_both_for_obj=True, scale_y=True,

                                                                    is_model_for_each_subject=model_for_each_subj,
                                                                    clip_drop_list=[], subj_drop_list=[],
                                                                    fs_models_list=FS_MODELS,
                                                                    fs_n_components_range=range(2, 10), pca_each_axis=pca_per_axis,
                                                                    learning_models_list=LEARNING_MODELS,
                                                                    ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
                                                                    is_second_learner=is_second_learner,

                                                                    is_majority_vote=is_majority_vote, scale_predicted_y_by=scale_predicted_y_by)

                                                        print("%s, smart=%s, seg_len=%i, ol=%s, %s, %s, model_for_each=%s, "
                                                              "2nd_learner=%s, majority_vote=%s, pca_per_axis=%s, scale_predicted_y_by=%s" %
                                                              (str(hl_margins).replace(', ', '-'), is_smart_hl, seg_len,
                                                               ol_percent, bs[1], obj_or_subj, model_for_each_subj, is_second_learner, is_majority_vote,
                                                               pca_per_axis, scale_predicted_y_by))

                                                        setwinsize = False
                                                        runsegmentize = False
                                                        runoverlap = False
                                                        runfeatures = False




        f.close()

    elif run_type == 'solo':

        # with open(LOG_FOLDER + 'log_' + datetime.now().strftime("%Y%m%d-%H%M%S_") + '.csv', 'w') as f:
            a = []

            for subj_details in [['336079314', 50, 50, False, 15], ['312282494', 50, 50, False, 15], ['301840336', 50, 50, False, 15],
                                 ['321720443', 50, 50, False, 15], ['203237607', 50, 50, False, 15], ['311357735', 50, 50, False, 15], ['203931639', 50, 50, False, 15]]:
            # for subj_details in [['336079314', 50, 50, False, 15]]:

                # [_, cur_result_wo, _, _], results_df, _, _ = mega_runner(str(run_id), f,
                [_, cur_result_wo, _, _], results_df, _, _ = mega_runner(str(run_id), None,
                            run_preprocessing=False, is_hl_in_preprocessing=False,

                            set_win_size=False, hl_margins=(5,1), is_smart_hl=True,

                            run_segmentize=False, is_hl=True, segments_length=subj_details[1],

                            run_overlap=False, overlap_percent=subj_details[2],

                            run_features=False, is_hl_in_features=True, create_moments_over_segmentized=False,   # when using not_hl, do create_moments_over_segmentized==True
                            is_slice_for_specific_blendshapes=True, which_blendshapes=MY_BS, use_overlap=True if subj_details[2] > 0 else False,

                            run_learning=True, obj_or_subj='obj', is_hl_in_learning=True,
                            is_both_for_obj=True, scale_y=True,

                            is_model_for_each_subject=True, clip_drop_list=[], subj_drop_list=[],
                            fs_models_list=FS_MODELS, fs_n_components_range=range(subj_details[4],subj_details[4]+1),

                            pca_each_axis=subj_details[3], learning_models_list=LEARNING_MODELS, ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
                            is_second_learner=True,

                            is_majority_vote=False, scale_predicted_y_by='org_clip',

                            # given_dfs=True, dfs=load_pickle_list_to_df(LOG_FOLDER + 'subj_dfs', subj_id_and_axis), subj_name=subj_name)
                            given_dfs=False, dfs=None)

                df_subj_slice = results_df.loc[results_df.index.get_level_values('subj_id') == subj_details[0]]
                r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) = calculate_corr(df_subj_slice.actual_y.tolist(), df_subj_slice.predicted_y.tolist(), method='normal')
                subjects_corr = [r2, pearsonr_val, pearsonr_p_val, spearman_val]

                a.append(subjects_corr[1])
            print(a)
                # f.write("%s, %s, %s, %s, %s, %s, %s\n" % (str(best_params[0]), str(best_params[1]), best_params[2], str(max_result), str(subjects_corr[1]), str(best_pca), subj_aside))
                # f.flush()
                # os.fsync(f)

                # results_df.to_csv('res_df_v_' + subj_details[0] + '.csv')

                # f.flush()
                # os.fsync(f)
        # f.close()


        # results_df.loc[results_df.index.get_level_values('subj_id') == cur_subj].to_csv('valence_results_df_solo.csv', mode='a', header='false')




    elif run_type == 'cross_validation':
        setwinsize = True
        with open(LOG_FOLDER + 'cv_log_' + datetime.now().strftime("%Y%m%d-%H%M%S_") + RATINGS_AXES[0][0].strip() + '.csv', 'w') as f:
            f.write("seg length, overlap, pca each axis, result without, results with\n")

            for subj_aside in dictionaries.SUBJECTS_IDS:        # leave subj aside
            # for subj_aside in dictionaries.SUBJECTS_IDS[::-1]:        # leave subj aside
            # for subj_aside in ['308476639']:
                max_result = 0
                best_pca = 0
                best_params = [None, None, None]

                for seg_len in [10,30,50]:
                # for seg_len in [10]:
                    runsegmentize = True
                    # for ol_percent in [25]:
                    for ol_percent in [0, 25, 50, 75]:
                        runoverlap = True if ol_percent > 0 else False
                        runfeatures = True
                        for pca_per_axis in [True, False]:
                        # for pca_per_axis in [True]:

                            # run full run with all parameter on 25 others (w.o. subj_aside)
                            [_, cur_result_wo, _, _], _, _, best_local_pca = mega_runner(str(run_id), None, run_preprocessing=False, is_hl_in_preprocessing=False,
                            set_win_size=setwinsize, hl_margins=(5,1), is_smart_hl=True,
                            run_segmentize=runsegmentize, is_hl=True, segments_length=seg_len,
                            run_overlap=runoverlap, overlap_percent=ol_percent,
                            run_features=runfeatures, is_hl_in_features=True, create_moments_over_segmentized=False,
                            is_slice_for_specific_blendshapes=True, which_blendshapes=MY_BS, use_overlap=True if ol_percent > 0 else False,
                            run_learning=True, obj_or_subj='obj', is_hl_in_learning=True,
                            is_both_for_obj=True, scale_y=True,
                            is_model_for_each_subject=True, clip_drop_list=[], subj_drop_list=[subj_aside],
                            fs_models_list=FS_MODELS, fs_n_components_range=range(2,20),
                            pca_each_axis=pca_per_axis, learning_models_list=LEARNING_MODELS, ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
                            is_second_learner=True, is_majority_vote=False, scale_predicted_y_by='org_clip')

                            runsegmentize = False
                            runoverlap = False
                            runfeatures = False
                            setwinsize = False

                            # take parameters from best run
                            print(cur_result_wo)
                            if cur_result_wo > max_result:
                                max_result = cur_result_wo
                                best_params = [seg_len, ol_percent, pca_per_axis]
                                best_pca = best_local_pca
                                print("new results won: " + str(max_result))

                # run full run with parameters - train set in 25 others and test is subj-aside
                print(",.-^-.,-^-.,-^-.,-^-.,-^-., Done Validating, Running Last Subject ,.-^-.,-^-.,-^-.,-^-.,-^-.,   " + subj_aside)
                print("with parameters: " + str(best_params) + ' ' + str(best_pca))

                _, results_df, [df_moments, df_quantized, df_dynamic, df_misc], _ = mega_runner(str(run_id), None, run_preprocessing=False, is_hl_in_preprocessing=False,
                set_win_size=True, hl_margins=(5,1), is_smart_hl=True,
                run_segmentize=True, is_hl=True, segments_length=best_params[0],
                run_overlap=True if best_params[1] > 0 else False, overlap_percent=best_params[1],
                run_features=True, is_hl_in_features=True, create_moments_over_segmentized=False,
                is_slice_for_specific_blendshapes=True, which_blendshapes=MY_BS, use_overlap=True,
                run_learning=True, obj_or_subj='subj', is_hl_in_learning=True,
                is_both_for_obj=True, scale_y=True,
                is_model_for_each_subject=True, clip_drop_list=[], subj_drop_list=[],
                fs_models_list=FS_MODELS, fs_n_components_range=range(best_pca,best_pca+1),     # good PCA!
                # fs_models_list=FS_MODELS, fs_n_components_range=range(2,20),                  # bad PCA!
                pca_each_axis=best_params[2], learning_models_list=LEARNING_MODELS, ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
                is_second_learner=False, is_majority_vote=False, scale_predicted_y_by='org_clip')

                df_subj_slice = results_df.loc[results_df.index.get_level_values('subj_id') == subj_aside]
                r2, (pearsonr_val, pearsonr_p_val), (spearman_val, spearman_p_val) = calculate_corr(df_subj_slice.actual_y.tolist(), df_subj_slice.predicted_y.tolist(), method='normal')
                subjects_corr = [r2, pearsonr_val, pearsonr_p_val, spearman_val]

                # write to file: parameters, results of 25-run and final result
                f.write("%s, %s, %s, %s, %s, %s, %s\n" % (str(best_params[0]), str(best_params[1]), best_params[2], str(max_result), str(subjects_corr[1]), str(best_pca), subj_aside))
                f.flush()
                os.fsync(f)

                fd = open(LOG_FOLDER + 'cv_results_df_' + RATINGS_AXES[0][0].strip() + '_' + str(run_id) + '.csv', 'a')
                fd.write('\n' + str(best_params[0]) + ' ' + str(best_params[1]) + ' ' + str(best_params[2]) + ' ' + str(max_result) + ' ' + str(subjects_corr[1]) + ' ' + subj_aside + '\n')
                fd.close()
                df_subj_slice.to_csv(LOG_FOLDER + 'cv_results_df_' + RATINGS_AXES[0][0].strip() + '_' + str(run_id) + '.csv', mode='a', header=False)

                for idx, df in enumerate([df_moments, df_quantized, df_dynamic, df_misc]):  # save dfs to reconstruct - give dfs straight to implicit_media_tagigng (if run isn't deterministic)
                    export_df_to_pickle(df, LOG_FOLDER + 'subj_dfs/' + subj_aside + '_' + RATINGS_AXES[0][0].strip() + '_' + str(run_id) + '_' + str(idx) + '.pkl')

                print(str(best_params[0]), str(best_params[1]), str(best_params[2]), str(max_result), str(subjects_corr[1]), subj_aside)

        f.close()