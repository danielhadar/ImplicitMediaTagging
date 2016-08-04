from preprocessing import *
from features import *
from learning import *
from time import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# LEARNING_MODELS = ['linear_regression', 'ridge', 'lasso', 'SVR', 'NuSVR']
# LEARNING_MODELS = ['linear_regression', 'ridge            ', 'SVR              ']
LEARNING_MODELS = ['linear_regression', 'ridge            ']
# LEARNING_MODELS = ['SVC']
# LEARNING_MODELS = ['SVR']
# LEARNING_MODELS = ['ridge']
# LEARNING_MODELS = ['linear_regression']

# CV_MODELS = ['none']
# CV_MODELS = ['loo']
# CV_MODELS = ['LeaveOneClipOutForEachSubject', 'LeaveOneClipOutForAllSubject', 'LeaveOneSubjOut']
# CV_MODELS = ['LeaveOneClipOutForEachSubject']
CV_MODELS = ['LeaveOneClipOutForAllSubject']
# CV_MODELS = ['LeaveOneSubjOut']


RATINGS_AXES = [('valence    ',0), ('arousal    ',1), ('likeability',2), ('rewatch    ',3)]
# RATINGS_AXES = [('likeability',2), ('rewatch    ',3)]
# RATINGS_AXES = [('rewatch',3)]

# FS_MODELS = ['none', 'pca']
FS_MODELS = ['pca']
# FS_MODELS = ['none']

def mega_runner(f, run_preprocessing, is_hl_in_preprocessing,
                set_win_size, hl_margins, is_smart_hl, run_segmentize, is_hl, segments_length,
                run_features, is_hl_in_features, create_moments_over_segmentized, is_slice_for_specific_blendshapes, which_blendshapes,
                run_learning, obj_or_subj, is_hl_in_learning, is_both_for_obj, scale_y, scale_x, use_single_predicted_Y_foreach_clip,
                is_model_for_each_subject, to_drop_list, fs_models_list, fs_n_components_range,
                learning_models_list, ratings_axes_list, cv_models_list, is_second_learner, is_majority_vote):
    f.write("Start Time: %s\n" % datetime.now().strftime("%H:%M:%S"))

    print(" > Starting Preprocessing...")
    if run_preprocessing:
        create_raw(is_hl=is_hl_in_preprocessing)
        f.write("run_preprocessing=True, with: is_hl=%s \n" % (is_hl_in_preprocessing))
    else:
        f.write("run_preprocessing=False \n")
    print("     Done Preprocessing!\n")


    print(" > Starting Set Window Size... ")
    if set_win_size:
        set_window_size(hl_margins, is_smart_hl)
        f.write("set_win_size=True, with: hl_margins=(%i, %i) \n" % (hl_margins[0], hl_margins[1]))
    else:
        f.write("set_win_size=False \n")
    print("     Done Set Window Size!\n")


    print(" > Starting Segmentizing... ")
    if run_segmentize:
        segmentize(seg_length=segments_length, is_hl=is_hl)
        f.write("run_segmentize=True, with: seg_len=%i \n" % (segments_length))
    else:
        f.write("run_segmentize=False \n")
    print("     Done Segmentizing!\n")


    print(" > Starting Creating Features... ")
    if run_features:
        create_features(use_hl=is_hl_in_features, slice_for_specific_bs=is_slice_for_specific_blendshapes, bs_list=which_blendshapes,
                        create_moments_over_segmentized=create_moments_over_segmentized)
        f.write("run_features=True, with: is_hl=%s , specific_blendshapes=%s \n" % (is_hl_in_features, is_slice_for_specific_blendshapes))
    else:
        f.write("run_features=False \n")
    print("     Done Creating Features!\n")


    print(" L e a r n i n g ... ")
    if run_learning:
        all_features_df, df_moments, df_quantized, df_dynamic, df_misc, objective_df, ratings_df, big5_df, raw_df, majority_objective_df = \
            learning_load_all_dfs(use_hl=is_hl_in_learning, use_both_for_obj=is_both_for_obj)
        y_df = objective_df if obj_or_subj == 'obj' else ratings_df
        if is_majority_vote:
            y_df = majority_objective_df
            ratings_axes_list = [('likeability',2), ('rewatch    ',3)]
            learning_models_list = ['SVC']

        elif scale_y:
            y_df = y_df.apply(scale)

        for cv_model_name in cv_models_list:
            for fs_model_name in fs_models_list:
                for axis in ratings_axes_list:
                    for fs_n_components in fs_n_components_range:
                        for learning_model_name in learning_models_list:

                            tic = time()
                            subjects_corr = implicit_media_tagging(df_moments, df_quantized, df_dynamic, df_misc, y_df, obj_or_subj=obj_or_subj,
                                                                   scale_x = scale_x, model_for_each_subject=is_model_for_each_subject,
                                                                   to_drop_list=to_drop_list, fs_model_name=fs_model_name,
                                                                   fs_n_components=fs_n_components, axis=axis,
                                                                   learning_model_name=learning_model_name, cv_model_name=cv_model_name,
                                                                   is_second_learner=is_second_learner,
                                                                   f=f, use_single_predicted_Y_foreach_clip=use_single_predicted_Y_foreach_clip)

                            f.write(learning_model_name + ', ' + cv_model_name + ', '
                                    + fs_model_name + str(fs_n_components) + ', ' + axis[0] + ', ' +
                                    str(scale_y) + ', ' + str(scale_x) + ', ' +
                                    str('%.3f' % np.mean([i[0] for i in subjects_corr])) + ', ' +
                                    str('%.3f' % np.mean([i[1] for i in subjects_corr])) + ', ' +
                                    str('%.3f' % np.mean([i[3] for i in subjects_corr])) + '\n')

                            toc = time()
                            print("%s, %s%.2i, %s, %s: %.3f, (p=%.3f), r^2=%.3f" %
                                  (learning_model_name, fs_model_name, fs_n_components, cv_model_name, axis[0],
                                   np.mean([i[0] for i in subjects_corr]), np.mean([i[1] for i in subjects_corr]),
                                   np.mean([i[3] for i in subjects_corr])))
                            # print(" In a total time of: %.2f" % (toc-tic))


if __name__ == '__main__':

    # with open('./logs/log_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', 'w') as f:
    #
    #     for gbs in [True, False]:
    #         for sl in [30,20,14,10,6]:
    #             for hlm in [(6,0), (6,1), (3,1), (5,0), (3,0), (5,1)]:
    #                 print("Starting: gbs=%s, sl=%i, hlm=%s" % (str(gbs), sl, hlm))
    #                 mega_runner(f, run_preprocessing=False, is_hl_in_preprocessing=True,
    #                     set_win_size=True, run_segmentize=True, is_hl=True, segments_length=30, hl_margins=(5,1),
    #                     run_features=False, is_hl_in_features=True, create_moments_over_segmentized=False,
    #                     is_slice_for_specific_blendshapes=gbs, which_blendshapes=GOOD_BLENDSHAPES,
    #                     run_learning=True, is_hl_in_learning=True,
    #                     is_both_for_obj=False, scale_y=True, scale_x=True, use_single_predicted_Y_foreach_clip=True,
    #                     is_model_for_each_subject=False, to_drop_list=[], fs_models_list=FS_MODELS, fs_n_components_range=range(2,10),
    #                     learning_models_list=LEARNING_MODELS, ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
    #                     is_second_learner=True)
    #
    # f.close()
    #
    #
    #
    mega_runner(open('dummy.csv', 'w'),
                run_preprocessing=False, is_hl_in_preprocessing=False,

                set_win_size=False, hl_margins=(5,1), is_smart_hl=False,

                run_segmentize=False, is_hl=True, segments_length=50,                                # to avoid hl start here

                run_features=True, is_hl_in_features=True, create_moments_over_segmentized=False,   # when using not_hl, do create_moments_over_segmentized==True
                is_slice_for_specific_blendshapes=True, which_blendshapes=MY_BS,

                run_learning=True, obj_or_subj='subj', is_hl_in_learning=True,
                is_both_for_obj=True, scale_y=False, scale_x=True, use_single_predicted_Y_foreach_clip=True,
                is_model_for_each_subject=False, to_drop_list=[], fs_models_list=FS_MODELS, fs_n_components_range=range(2,8),
                learning_models_list=LEARNING_MODELS, ratings_axes_list=RATINGS_AXES, cv_models_list=CV_MODELS,
                is_second_learner=False, is_majority_vote=False)                                     # to use is_majority_vote set obj_or_subj='obj'