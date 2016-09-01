import numpy as np
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from math import isnan
from scipy import cluster
from sklearn.decomposition import PCA
from sklearn.cross_validation import LeavePLabelOut
import pandas as pd

global SUBJECTS_IDS

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

fps = 24
NUM_CLIPS = 18

PARENT_FOLDER = '/Volumes/MyPassport/phase_b/'      # change here where changing machine
# PARENT_FOLDER = '/cs/img/danielhadar/'

RATINGS_DIR = PARENT_FOLDER + 'subjects_ratings/'
DATA_FOLDER = PARENT_FOLDER + 'raw_and_rest_data/'
PICKLES_FOLDER = PARENT_FOLDER + 'pickles/'
CSV_FOLDER = PARENT_FOLDER + 'csv/'
LOG_FOLDER = PARENT_FOLDER + 'logs/'
OBJECTIVE_FOLDER = '/Users/danielhadar/Documents/Thesis/ExperimentCode/metadata/subject rating PhaseA'

BLENDSHAPES = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L', 'EyeIn_R',
               'EyeOpen_L', 'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R',
               'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen', 'LipsTogether', 'JawLeft', 'JawRight', 'JawFwd',
               'LipsUpperUp_L', 'LipsUpperUp_R', 'LipsLowerDown_L', 'LipsLowerDown_R', 'LipsUpperClose',
               'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L',
               'LipsStretch_R', 'MouthFrown_L', 'MouthFrown_R', 'MouthPress_L', 'MouthPress_R', 'LipsPucker',
               'LipsFunnel', 'MouthLeft', 'MouthRight', 'ChinLowerRaise', 'ChinUpperRaise', 'Sneer_L', 'Sneer_R',
               'Puff', 'CheekSquint_L', 'CheekSquint_R']    # len = 51

GOOD_BLENDSHAPES = ['EyeBlink_L', 'EyeBlink_R','EyeIn_L', 'EyeIn_R', 'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen', 'MouthLeft',
                    'MouthRight', 'MouthFrown_L', 'MouthFrown_R', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L',
                    'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R', 'LipsUpperUp', 'LipsFunnel', 'ChinLowerRaise',
                    'Sneer', 'CheekSquint_L', 'CheekSquint_R']      # len = 24

MY_BS = ['EyeBlink_L', 'EyeBlink_R', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L',
               'LipsStretch_R', 'Sneer_L', 'Sneer_R']

SUBJECTS_DICT = {315823492: [6, 10], 315688713: [4, 10], 337835383: [8, 11], 200398733: [6, 15],
                 308286285: [6, 9], 301840336: [4, 14], 336079314: [4, 11], 203667092: [11, 11],
                 304957913: [5, 9], 304854938: [4, 10], 311461917: [5, 10], 203712351: [5, 15],
                 304835366: [5, 10], 332521830: [11, 15], 203237607: [4, 17], 311357735: [5, 14],
                 305584989: [12, 16], 308476639: [5, 14], 204033971: [4, 9], 312282494: [4, 10],
                 203931639: [5, 24], 204713721: [4, 15], 321720443: [6, 15], 317857084: [5, 10],
                 204058010: [5, 10], 203025663: [5, 9]}     # len = 26

MOUTH_BS = ['JawOpen', 'LipsTogether', 'LipsUpperUp_L', 'LipsUpperUp_R', 'LipsLowerDown_L', 'LipsLowerDown_R', 'LipsUpperClose',
               'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L',
               'LipsStretch_R', 'MouthFrown_L', 'MouthFrown_R', 'MouthPress_L', 'MouthPress_R', 'LipsPucker',
               'LipsFunnel', 'MouthLeft', 'MouthRight', 'ChinLowerRaise', 'ChinUpperRaise']

EYES_AREA_BS = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'BrowsD_L', 'BrowsD_R',
               'BrowsU_C', 'BrowsU_L', 'BrowsU_R']

SMILE_BS = ['MouthSmile_L', 'MouthSmile_R']

BLINKS_BS = ['EyeBlink_L', 'EyeBlink_R']

inf = float('Inf')

def flatten_list(l):
    if np.ndim(l) == 1:
        return l
    return [float(item) for sublist in l for item in sublist]


def slice_features_df_for_specific_blendshapes(df, blendshapes_list):

    new_columns_list = []

    for col in df.columns.values:
        if col == 'time':
            new_columns_list.append('time')
        elif col == 'is_locked':
            new_columns_list.append('is_locked')
        elif col == 'ind':
            new_columns_list.append('ind')
        elif col == 'new_response_type':
            new_columns_list.append('new_response_type')

        else:
            for b in blendshapes_list:
                if b in col:
                    new_columns_list.append(col)
                    break

    return df.ix[:,new_columns_list].copy()

def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot

def scale_list(l):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(l)-min(l) == 0:
        return l
    for idx,val in enumerate(l):
        l[idx] = ((val - min(l)) / (max(l)-min(l))) * (top-bot) + bot
    return l

def my_pow(val):
    return val**2

def unique_sequences_in_list(l):
    # [2,1,2,3,3,1,1,1,3,3,3,1] -> [2, 1, 2, 3, 1, 3, 1]
    ret_list = [l[0]]

    for i in range(1, len(l)):
        if l[i] == l[i-1]:
            continue
        else:
            ret_list.append(l[i])

    return ret_list

def list_to_unique_list_preserve_order(list):
    # http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [i for i in list if not (i in seen or seen_add(i))]

def grouper(iterable, n, fillvalue=None):
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def export_df_to_pickle(df, export_path):
    df.to_pickle(export_path)

def load_pickle_to_df(import_path):
    import pandas as pd
    return pd.read_pickle(import_path)

def export_dict_to_pickle(dict, export_path):
    import pickle
    with open(export_path, 'wb') as handle:
        pickle.dump(dict, handle)

def load_pickle_to_dict(import_path):
    import pickle
    with open(import_path, 'rb') as handle:
        return pickle.load(handle)


def find_peaks(v, delta=0.1, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta < 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
            # if this < delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
            # if this > delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    # return np.array(maxtab), np.array(mintab)
    return np.array(maxtab)

def count_peaks(v, delta=0.1, x = None):
    return len(find_peaks(v, delta, x))

def get_majoity(list, bin=False, th=2):
    # returns the majority vote.
    # bin=True yields a binary list (2-class: 0/1) - i.e. thresholding
    if bin:
        return 1 if np.mean(list) > th else 0
    else:
        return max(set(list), key=list.count)

def get_pos_or_neg(list):
    # returns a quantized yes/no (for rewatch, likeability) - 0 or 1
    return_list = []
    for i in list:
        if i >= 2:
            return_list.append(1.0)
        else:
            return_list.append(0.0)
    return return_list

def se_of_regression(actual, predicted):
    # Standard error of the regression
    # https://www.otexts.org/fpp/4/4
    N = len(predicted)
    return np.sqrt(
        (1/(N-2)) * sum([pow((a-b),2) for a,b in zip(predicted, actual)])
    )

def balanced_accuracy_score(actual, predicted):
    # balanced accuracy score: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    # (TP/P + TN/N)/2
    # matrix is: [[TN,FN],[FP,TP]]
    from sklearn.metrics import confusion_matrix
    [[tn, fn],[fp, tp]] = confusion_matrix(predicted, actual)

    return (tp/(tp+fp) + tn/(tn+fn))/2

def my_kmeans(col, n_quants=4, quantization_method='random'):
    # implemented so quantization could be done using apply
    return cluster.vq.kmeans2(col.values, n_quants, thresh=1e-02, minit=quantization_method, missing='warn')[1]

def previous_and_next_iterator(iterable):
    # iterates over 'iterable' while allowing access to prev and next
    # http://stackoverflow.com/questions/1011938/python-previous-and-next-values-inside-a-loop
    # modified for df.groupby, changed None of beginning and end to tuples of (None,None)
    from itertools import tee, islice, chain

    prevs, items, nexts = tee(iterable, 3)
    prevs = chain([(None, None)], prevs)
    nexts = chain(islice(nexts, 1, None), [(None, None)])
    return zip(prevs, items, nexts)

def slip_by_underline(str):
    return str.split('_')[0]

def add_original_clip(df):
    # adds index column 'org_clip' based upon index column 'clip_id' and return ['subj_id', 'clip_id', 'org_clip']
    df['org_clip'] = df.index.get_level_values('clip_id')
    df['org_clip'] = df['org_clip'].apply(lambda x: int(x.split('_')[0]))
    return df.set_index('org_clip', append=True).reorder_levels(['subj_id', 'clip_id', 'org_clip'])

def add_y(df, y_df, axis):
    # adds y ratings to the df
    df = df.reset_index('clip_id')
    for clip in y_df.index:
        df.loc[clip, axis[0].strip()] = y_df.loc[clip, axis[0].strip()]
    return df.set_index('clip_id', append=True)

def scale_column_by(df, column_name, scale_by):
    df[column_name] = df.groupby(level=scale_by)[column_name].apply(scale)
    return df

if __name__ == '__main__':
    mylist = ['banana', 'orange', 'apple', 'kiwi', 'tomato']

    for previous, item, nxt in previous_and_next_iterator(mylist):
        print("Item is now", item, "next is", nxt, "previous is", previous)