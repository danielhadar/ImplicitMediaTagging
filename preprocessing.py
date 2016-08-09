import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl
import pickle
from features import *
from utils import *


def draw_heatmap(df):
    t = df[df.clip_id == 5].loc[:,'time'].tolist()

    for idx, field in enumerate(BLENDSHAPES):
        s = df[df.clip_id == 5].loc[:,field].tolist()
        plt.subplot(len(BLENDSHAPES), 1, idx+1)
        plt.plot(t, s)


    plt.show()

# -----------------------------------
# --- Objective Ratings (Phase A) ---
# -----------------------------------
# This one works for all subjects together

def get_objective_ratings(which_phase='just_a'):
    if which_phase=='just_a':
        return pd.read_csv(OBJECTIVE_FOLDER + '/__clips_averages_for_phaseB_from_phaseA.csv', index_col=0)

    elif which_phase == 'both':
        return pd.read_csv(OBJECTIVE_FOLDER + '/__clips_averages_for_phaseB_from_phaseAandB.csv', index_col=0)

# --------------------------------
# --- Subject's Ratings Handle ---
# --------------------------------
# This one works for all subjects together

def get_ratings_data_and_create_df_for_all_subjects():
    list_of_all_dfs_to_concat = []

    for root, subj_dir, files in os.walk(RATINGS_DIR):
        for name in files:
            if name.endswith('.xlsx'):
                # A:F is to read V,A,L,R,F with clip_id as index
                xl_file = pd.read_excel(os.path.join(root, name), parse_cols="A:F", index_col=0)
                xl_file = xl_file[:18]    # take only the rows with ratings
                # add index column with subj_id
                xl_file['subj_id'] = name[:9]
                xl_file.set_index('subj_id', append=True, inplace=True)
                xl_file = xl_file.reorder_levels(['subj_id', 'clip_id'])

                list_of_all_dfs_to_concat.append(xl_file)

    return pd.concat([df for df in list_of_all_dfs_to_concat])


# --------------------------------
# ---  Subject's Big-5 Handle  ---
# --------------------------------
# This one works for all subjects together

def calculate_big5_and_create_df_for_all_subjects():
    list_of_all_dfs_to_concat = []

    for root, subj_dir, files in os.walk(RATINGS_DIR):
        for name in files:
            if name.endswith('.xlsx'):
                wb = openpyxl.load_workbook(os.path.join(root, name))
                cur_sheet = wb.get_sheet_by_name('Sheet1')
                cur_big_5_raw = np.array([cur_sheet.cell(row=i, column=7).value for i in range(1,45)])

                for i in [1, 5, 7, 8, 11, 17, 20, 22, 23, 26, 30, 33, 34, 36, 40, 42]:
                    cur_big_5_raw[i] = 5-cur_big_5_raw[i]

                cur_df = pd.DataFrame({
                    'extraversion':np.mean(cur_big_5_raw[[0, 5, 10, 15, 20, 25, 30, 35]]),
                    'neuroticism':np.mean(cur_big_5_raw[[3, 8, 13, 18, 23, 28, 33, 38]]),
                    'agreeableness':np.mean(cur_big_5_raw[[1, 6, 11, 16, 21, 26, 31, 36, 41]]),
                    'concientiousness':np.mean(cur_big_5_raw[[2, 7, 12, 17, 22, 27, 32, 37, 42]]),
                    'openness_to_experience':np.mean(cur_big_5_raw[[4, 9, 14, 19, 24, 29, 34, 39, 40, 43]])
                }, index=[name[:9]])

                list_of_all_dfs_to_concat.append(cur_df)

    df = pd.concat([df for df in list_of_all_dfs_to_concat])
    df.index.name = 'subj_id'

    return df


# --------------------------------
# ---       Rest Handle        ---
# --------------------------------
# This one works for each subject separately, by changing CUR_SUBJ id and REST_T0

def get_resting_state_vid_to_df(subj_id, rest_t0):
    # export_path: PICKLES_FOLDER+'/'+CUR_SUBJ+'_rest.pickle'

    # get all columns and data
    full_rest_df = pd.read_csv(DATA_FOLDER+'/'+str(subj_id)+'_rest.txt',
                sep=' ',
                names=['c'+str(i) for i in range(178)],
                index_col=False
    )

    # slice relevant columns
    fixed_rest_df = pd.DataFrame(
        data=full_rest_df.loc[:,['c3','c4']+['c'+str(i) for i in range(15,66)]].values,
        columns=['time', 'is_locked'] + BLENDSHAPES
    )

    # remove irrelevant times and calibrate according to T0
    fixed_rest_df['time'] /= 1000
    fixed_rest_df = fixed_rest_df[fixed_rest_df.time > rest_t0+1]       # start count after REST_T0 (beep)
    fixed_rest_df['time'] -= min(fixed_rest_df.time.tolist())
    fixed_rest_df = fixed_rest_df[fixed_rest_df.time < 60]    # end count after 1 minute

    return fixed_rest_df.set_index('time')

# --------------------------------
# ---       Raw Handle         ---
# --------------------------------
# This one works for each subject separately, by changing CUR_SUBJ id and RAW_T0

def get_clips_and_times(subj_id):
    # read all clips and times
    clips_df = pd.read_csv(RATINGS_DIR+'/'+str(subj_id)+'/'+str(subj_id)+'.txt',
                        names=['clip_id','start_time','end_time','start_verbal','end_verbal'],
                        index_col=0)

    # calibrate clips time according to beep (make t=0 be the beginning of first clip [i.e. after beep+fixation_period])
    t0 = min(clips_df['start_time'].tolist())
    for field in ['start_time','end_time','start_verbal','end_verbal']:
        clips_df[field] -= t0

    clips = clips_df.apply(np.ceil, axis=1)    # round UP
    # end_time = max(clips_df['end_time'].tolist())

    return clips_df

def get_raw_vid_to_df(subj_id, raw_t0):
    # read subj_raw.txt
    raw_df = pd.read_csv(DATA_FOLDER+'/'+str(subj_id)+'_raw.txt',
                     sep=' ',
                     names=['c'+str(i) for i in range(178)],
                     index_col=False)

    # take from raw_df: time, is_locked, blendshapes
    df = pd.DataFrame(
        data=raw_df.loc[:,['c3','c4']+['c'+str(i) for i in range(15,66)]].values,
        columns=['time', 'is_locked'] + BLENDSHAPES
    )

    # remove irrelevant times and calibrate according to T0
    df['time'] /= 1000
    df = df[df.time > raw_t0]       # start count after RAW_t0 (beep)
    df['time'] -= min(df.time.tolist())

    # add subj_id column
    df['subj_id'] = str(subj_id)

    return df


def add_response_type_and_clip_id_to_raw_df(df, clips_df, use_hl=False, to_segmentize=False, seg_length_in_frames=12,
                                            hl_margins=(5, 1)):
    # response types: watch, verbal, none.
    # clip is considered to start clips start, and to end at the verbal report.
    # use_hl == should use Highlight? i.e., the response types are: watch, verbal, none AND hl. hl is taken from
    #                                           watch-HL_MARGINS until watch+HL_MARGINS.

    # add columns: response_type, clip_id
    df['response_type'] = 'none'    # create new column
    df['clip_id'] = -1              # create new column


    # iterate along the df and add clips_ids and response times:

    cur_clip_idx = 0    # hold pointer to current clip *index* (not id)
    is_between_clips = False

    cur_clip_id = clips_df.index.values[cur_clip_idx]   # current clip *id*
    cur_clip_df_entry = clips_df.loc[cur_clip_id, :]    # current clip df entry (times)

    # designated for segmentizing
    seg_num = 1
    frame_count = 0     # when reached frame_count==seg_length_in_frames, increase current segment number
    not_first = False

    # iterate along all rows in the df:
    for idx, row in df.iterrows():

        # if current row time is within clip *total* time
        if cur_clip_df_entry['start_time'] <= row['time'] <= cur_clip_df_entry['end_verbal']:
            is_between_clips = False
            df.loc[idx, 'clip_id'] = str(cur_clip_id)

            if use_hl:
                # if current row time is within *watching* time-HL_MARGINS
                if row['time'] <= (cur_clip_df_entry['end_time']-hl_margins[0]):
                    df.loc[idx, 'response_type'] = 'watch'

                # if current row time is after watch-HL_MARGINS but before watch+HL_MARGINS
                elif row['time'] <= (cur_clip_df_entry['end_time']+hl_margins[1]):
                    df.loc[idx, 'response_type'] = 'hl'

                    if to_segmentize:
                        df.loc[idx, 'clip_id'] = str(str(cur_clip_id) + '_' + str(seg_num))
                        frame_count += 1
                        if frame_count == seg_length_in_frames:
                            frame_count = 0
                            seg_num += 1
                        if frame_count > 1:
                            not_first = True


                # if current row time is within *verbal* time (and yet after watch+HL_MARGINS)
                elif row['time'] >= cur_clip_df_entry['start_verbal']:
                    df.loc[idx, 'response_type'] = 'verbal'

                # current row is after end_time (watch) but before start_verbal
                else:
                    df.loc[idx, 'response_type'] = 'none'

                # designated for edge case where a segments contains only one frame; that causes NaNs when computing
                #           moments. Therefore, segments like this are removed
                if not_first:
                    if (df.loc[[idx-1]].response_type.values == 'hl' and df.loc[[idx]].response_type.values != 'hl')\
                        and (df.loc[[idx-1]].clip_id.values != df.loc[[idx-2]].clip_id.values):
                        df.drop(idx-1, inplace=True)

            else:
                # if current row time is within *watching* time
                if row['time'] <= cur_clip_df_entry['end_time']:
                    df.loc[idx, 'response_type'] = 'watch'

                    if to_segmentize:
                        df.loc[idx, 'clip_id'] = str(str(cur_clip_id) + '_' + str(seg_num))
                        frame_count += 1
                        if frame_count == seg_length_in_frames:
                            frame_count = 1
                            seg_num += 1

                # if current row time is within *verbal* time
                elif row['time'] >= cur_clip_df_entry['start_verbal']:
                    df.loc[idx, 'response_type'] = 'verbal'

                # current row is after end_time (watch) but before start_verbal
                else:
                    df.loc[idx, 'response_type'] = 'none'

        # current row time is between clips (after verbal_end and before start_time)
        # increase clip_idx by 1 and "skip" all rows until the new start_time
        elif not is_between_clips:
            cur_clip_idx += 1        # pointer to current clip *index* (not id)
            seg_num = 1
            frame_count = 1

            # if reached last clip, finish for this subject
            if cur_clip_idx >= len(clips_df.index):
                break

            cur_clip_id = clips_df.index.values[cur_clip_idx]   # current clip *id*
            cur_clip_df_entry = clips_df.loc[cur_clip_id, :]    # current clip df entry (times)
            is_between_clips = True

    # Re-index: subj_id, clip_id, response_type
    df.set_index(['subj_id', 'clip_id', 'response_type'], inplace=True)
    df = df.reorder_levels(['subj_id', 'clip_id', 'response_type'])

    df.to_csv('a.csv')

    return df



# --------------------------------
# ---     Helper Functions     ---
# --------------------------------

def stitch_all_dfs_in_folder_together(dfs_path):
    list_of_all_dfs_to_concat = []

    i = 1
    for file in os.listdir(dfs_path):
        if file == ".DS_Store":
            continue

        cur_temp_df = load_pickle_to_df( os.path.join(dfs_path, file) )
        list_of_all_dfs_to_concat.append(cur_temp_df)
        # print(str(i) + ' ' + file)
        i += 1

    return pd.concat([d for d in list_of_all_dfs_to_concat])

# ----------
# -- MAIN --
# ----------

def create_raw(is_hl):

    for subj_id, t0s in SUBJECTS_DICT.items():
        print("Pre-processing: " + str(subj_id))

        clips_df = get_clips_and_times(subj_id)
        subj_df = get_raw_vid_to_df(subj_id, t0s[1])

        if is_hl:
            df = add_response_type_and_clip_id_to_raw_df(subj_df, clips_df, use_hl=True,
                                                     to_segmentize=False, seg_length_in_frames=None,
                                                     hl_margins=None)
            export_df_to_pickle(df, PICKLES_FOLDER + '/raw/with_hl/' + str(subj_id) + '_raw.pickle')

            # commented out, but could be used to compute moments over unsegmentized raw
            # df_unsegmentized = add_response_type_and_clip_id_to_raw_df(subj_df, clips_df, use_hl=True,
            #                                          to_segmentize=False, seg_length_in_frames=seg_len,
            #                                          hl_margins=hl_margins)
            # export_df_to_pickle(df_unsegmentized, PICKLES_FOLDER + '/raw/with_hl/' + str(subj_id) + '_raw_unsegmentized.pickle')
        else:
            df = add_response_type_and_clip_id_to_raw_df(subj_df, clips_df,
                                                         to_segmentize=False, seg_length_in_frames=None,
                                                         hl_margins=None)
            export_df_to_pickle(df, PICKLES_FOLDER + '/raw/wo_hl/' + str(subj_id) + '_raw.pickle')

            # df_unsegmentized = add_response_type_and_clip_id_to_raw_df(subj_df, clips_df,
            #                                              to_segmentize=False, seg_length_in_frames=seg_len,
            #                                              hl_margins=hl_margins)
            # export_df_to_pickle(df_unsegmentized, PICKLES_FOLDER + '/raw/wo_hl/' + str(subj_id) + '_raw_unsegmentized.pickle')

    # > Stitch all raw DFs
    print(" ... Stiching all raw DFs ...")
    if is_hl:
        df = stitch_all_dfs_in_folder_together('/Volumes/MyPassport/phase_b/pickles/raw/with_hl')
        export_df_to_pickle(df, PICKLES_FOLDER + '/org_raw_with_hl.pickle')
    else:
        df = stitch_all_dfs_in_folder_together('/Volumes/MyPassport/phase_b/pickles/raw/wo_hl')
        export_df_to_pickle(df, PICKLES_FOLDER + '/org_raw.pickle')


def get_smart_hl_pivot_ind(data, hl_margins):
    # return the index (in terms of ind) with the most interesting environment
    # data is the 'watch' part of a single subject's single clip
    hl_win_size = sum(hl_margins)

    # blink frequency
    # smile left (36)
    # smile right (37)
    # mouth dimple left (38)
    # mouth dimple right (39)
    # lips strech left (40)
    # lips strech right (41)
    # mouth frown left (42)
    # mouth frown right (43)
    # data.to_csv('d.csv')
    # quit()

    # for each possible window:
    #   sum(for each au, average it's value)
    clip_len = int(data['time'].tail(1).values-data['time'].head(1).values)  # in seconds
    l = [0] * (clip_len - hl_win_size + 1)                                   # amount of possible hl windows

    # df of windows properties
    windows_df = pd.DataFrame({'blinks': l, 'smile': l, 'mouth_dimple': l,
                               '_begin_frame':[i*fps for i in range(len(l))],
                               '_end_frame':[(i+hl_win_size)*fps for i in range(len(l))]})
    windows_df[['mouth_dimple','smile']] = windows_df[['mouth_dimple','smile']].astype(float)

    # create a list of times of blinks (in frames-times)
    blink_times_L = [i[0] for i in find_peaks(data['EyeBlink_L'], delta=data['EyeBlink_L'].mean()/2)]
    blink_times_R = [i[0] for i in find_peaks(data['EyeBlink_R'], delta=data['EyeBlink_R'].mean()/2)]
    blink_times = blink_times_L if (len(blink_times_L) > len(blink_times_R)) else blink_times_R

    for idx, row in windows_df.iterrows():
        windows_df.set_value(idx, 'blinks', len([t for t in blink_times if row._end_frame >= t >= row._begin_frame]))
        windows_df.set_value(idx, 'mouth_dimple',
                             np.mean((data.iloc[int(row._begin_frame):int(row._end_frame)].loc[:,'MouthDimple_L'].mean(),
                                      data.iloc[int(row._begin_frame):int(row._end_frame)].loc[:,'MouthDimple_R'].mean())))
        windows_df.set_value(idx, 'smile',
                             np.mean((data.iloc[int(row._begin_frame):int(row._end_frame)].loc[:,'MouthSmile_L'].mean(),
                                      data.iloc[int(row._begin_frame):int(row._end_frame)].loc[:,'MouthSmile_R'].mean())))


    windows_df.ix[:,'mouth_dimple':] = windows_df.ix[:,'mouth_dimple':].apply(scale)
    windows_df['summ'] = windows_df.iloc[:,2:].sum(axis=1)

    return int(windows_df.loc[[windows_df.summ.idxmax()]]._begin_frame \
           + fps*hl_margins[0] + int(data.ind.head(1)))             # best frame + seconds from beginning + start idx

    # return idx + hl_margins[0]

    # jaw open? (25)
    # lips together (26)
    # jaw left (27)
    # jaw right(28)
    # jaw fwd (29)
    # lip upper up left (30)
    # lip upper up right (31)
    # lip upper close (34)
    # lip lower close (35)




    # for (t, bt, sm, sv) in a:
    #     print("%i, %i, %.2f, %.2f" % (t, bt, sm, sv))
    # for i in range(len(a)):
    #     print("%i, %.2f, %.2f, %.2f" % (a[i][0], bts[i], sms[i], svs[i]))





def set_window_size(hl_margins, is_smart_hl):
    # set 'hl' section

    ratings_df, big5_df, objective_df, raw_df, hl_df = load_all_dfs(org=True)

    # this column was added to addressing 'row_idx' would be unique (id, hl, clip_id, ind)
    raw_df['ind'] = range(1, len(raw_df) + 1)
    raw_df['new_response_type'] = raw_df.index.get_level_values('response_type')

    # scale each AU column between 0-1
    # raw_df.ix[:,'EyeBlink_L':'CheekSquint_R'] = raw_df.ix[:,'EyeBlink_L':'CheekSquint_R'].apply(scale)

    # for idx, data in raw_df.groupby(level=[0,1,2], sort=False):
    for idx, data in raw_df[raw_df.new_response_type == 'watch'].groupby(level=[0,1,2], sort=False):
        # if idx[0] != '200398733':
        # if idx != ('336079314', '10', 'watch'):
        # if idx != ('203025663', '9', 'watch'):
        #     continue

        # gets the index to build the hl window around it
        hl_pivot_ind = get_smart_hl_pivot_ind(data, hl_margins) \
            if is_smart_hl else int(data.iloc[[-1]].ind)

        clip_start_hl_ind = hl_pivot_ind - hl_margins[0]*fps
        clip_end_hl_ind = hl_pivot_ind + hl_margins[1]*fps

        raw_df.ix[clip_start_hl_ind:clip_end_hl_ind,'new_response_type'] = 'hl'

    # fix columns:
    raw_df.reset_index(level=['response_type'], inplace=True)
    raw_df.drop(['response_type','ind'], axis=1, inplace=True)
    raw_df.rename(columns={'new_response_type':'response_type'}, inplace=True)
    raw_df.set_index(['response_type'], append=True, inplace=True)
    raw_df = raw_df.reorder_levels(['subj_id', 'clip_id', 'response_type'])

    export_df_to_pickle(raw_df, PICKLES_FOLDER + '/org_raw_with_hl.pickle')

def segmentize(seg_length, is_hl=True):

    ratings_df, big5_df, objective_df, raw_df, hl_df = load_all_dfs(org=True)

    if is_hl:
        df = hl_df
        work_on = 'hl'
    else:
        df = raw_df
        work_on = 'watch'

    # change name of clip_id to original_clip and add column called 'clip_id'. currently are identical.
    df.reset_index(level=['clip_id'], inplace=True)
    df.rename(columns={'clip_id':'original_clip'}, inplace=True)
    df['clip_id'] = df['original_clip']
    df.set_index(['original_clip'], append=True, inplace=True)

    # this column was added to addressing 'row_idx' would be unique (id, hl, clip_id, ind)
    df['ind'] = range(1, len(df) + 1)
    df.set_index(['ind'], append=True, inplace=True)

    cur_subj_id = [0, 0]
    for idx, data in df.groupby(level=[0,1,2], sort=False):

        if idx[1] != work_on:   # work only on relevant response types
            continue

        if idx[0] != cur_subj_id[1]:
            cur_subj_id[1] = idx[0]
            cur_subj_id[0] += 1
            print(cur_subj_id)

        cur_clip_id = idx[2]

        amount_of_seg, reminder = divmod(len(data),seg_length)
        for i in range(amount_of_seg):
            data.ix[(i*seg_length):(((i+1)*seg_length)),'clip_id'] = str(str(cur_clip_id) + '_' + str(i+1))

        # this if is for cases for last segment with 1 frame (later var==nan). adds this lonely frame to previous ss
        if len(data.ix[((i+1)*seg_length):len(data),'clip_id']) == 1:
            data.ix[((i+1)*seg_length):len(data),'clip_id'] = str(str(cur_clip_id) + '_' + str(i+1))
        else:
            data.ix[((i+1)*seg_length):len(data),'clip_id'] = str(str(cur_clip_id) + '_' + str(i+2))

        df.loc[idx] = data

    # fix columns:
    df.reset_index(level=['original_clip','ind'], inplace=True)
    df.drop(['original_clip','ind'], axis=1, inplace=True)
    df.set_index(['clip_id'], append=True, inplace=True)
    df = df.reorder_levels(['subj_id', 'clip_id', 'response_type'])

    # export
    if is_hl:
        export_df_to_pickle(df, PICKLES_FOLDER + '/raw_with_hl.pickle')
    else:
        export_df_to_pickle(df, PICKLES_FOLDER + '/raw.pickle')



if __name__ == '__main__':

    # --- RUN ONLY ONCE: create all pickles ---

    #   > SUBJECTIVE RATINGS
    #                    valence  arousal  like  rewatch  familiarity
    # subj_id   clip_id
    # 200398733 9              3        3     1        1            1
    #
    # df = get_ratings_data_and_create_df_for_all_subjects()
    # export_df_to_pickle(df, PICKLES_FOLDER + '/ratings.pickle')

    # This code segment was used to create ratings averages csv table to create objective-ratings-phaseA+B. This code
    #   creates averages/std just for phaseB, and phaseA was later added manually in excel:
    # print(df.groupby(level=[1]).std().to_csv())


    #   > OBJECTIVE RATINGS
    #          valence  arousal  likeability  rewatch  faimliarity  Valence_STD  ...
    # clip_id
    # 1           4.04     2.96         2.62     2.27         1.04         1.00
    # df = get_objective_ratings(which_phase='both')
    # export_df_to_pickle(df, PICKLES_FOLDER + '/objective_both.pickle')
    # df = get_objective_ratings(which_phase='just_a')
    # export_df_to_pickle(df, PICKLES_FOLDER + '/objective.pickle')


    #   > BIG-5
    #            agreeableness  concientiousness  extraversion  neuroticism  openness_to_experience
    # subj_id
    # 200398733       3.888889          3.444444         2.875         2.25        3.8
    #
    # df = calculate_big5_and_create_df_for_all_subjects()
    # export_df_to_pickle(df, PICKLES_FOLDER + '/big5.pickle')


    #   > RESTING and RAW
    #              is_locked  EyeBlink_L  EyeBlink_R  EyeSquint_L  EyeSquint_R  ...  all AUs
    # time
    # 0.00000          1    0.090571    0.137663     0.015925     0.071698      ...
    # 0.04166          1    0.070108    0.134684     0.041386     0.081688      ...
    #
    #                                        time  is_locked  EyeBlink_L  EyeBlink_R  ... all AUs
    # subj_id   clip_id response_type
    # 311461917 9       watch          0.0000          1    0.092798    0.154438
    #                   watch          0.0416          1    0.100907    0.161763
    #
    # for subj_id, t0s in SUBJECTS_DICT.items():
    #     print("Pre-processing: " + str(subj_id))
    #     # resting
    #     df = get_resting_state_vid_to_df(subj_id, t0s[0])   # subj_id and rest_T0
    #     export_df_to_pickle(df, PICKLES_FOLDER + '/rest/' + str(subj_id) + '_rest.pickle')
    #
    # raw
    # create_raw(is_hl=True)

    #   > Majority Vote subjects ratings

    df = load_pickle_to_df(PICKLES_FOLDER + '/ratings.pickle')
    majority_objective_df = load_pickle_to_df(PICKLES_FOLDER + '/objective.pickle')
    majority_objective_df.drop(majority_objective_df.columns[[0,1,4,5,6,7,8,9]], axis=1, inplace=True)
    for idx, data in df.groupby(level=1):
        majority_objective_df.loc[idx, 'rewatch'] = get_majoity(data.rewatch.tolist(), bin=True, th=df.rewatch.mean())
        majority_objective_df.loc[idx, 'likeability'] = get_majoity(data.likeability.tolist(), bin=True, th=df.likeability.mean())
    export_df_to_pickle(majority_objective_df, PICKLES_FOLDER + '/majority_objective.pickle')

    # --- RUN ONLY ONCE: end ---



    print(" > Done Pre-processing")
