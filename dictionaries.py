# from utils import PARENT_FOLDER

SUBJECTS_IDS = ['311461917', '304835366', '304854938', '315688713', '204058010', '203237607', '204033971', '203025663',
                '203931639', '308286285', '315823492', '203667092', '312282494', '304957913', '200398733', '332521830',
                '336079314', '317857084', '311357735', '204713721', '337835383', '203712351', '305584989', '308476639',
                '301840336', '321720443']    # len = 26

CLIPS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 15, 18, 24, 28, 29, 33, 34, 35, 36]   # len = 18

PICKLES_FOLDER = '/cs/img/danielhadar/' + 'pickles'

CLIPS_AND_TIMES = {'15': 23, '29': 20, '7': 19, '28': 25, '35': 22, '33': 20, '4': 13, '18': 29, '12': 19, '1': 16,
                   '10': 23, '2': 30, '36': 20, '6': 12, '24': 22, '9': 23, '5': 6, '34': 19}

if __name__ == '__main__':
    import learning
    import numpy as np
    import utils
    from scipy.stats import pearsonr, spearmanr
    from math import isnan

    subjective_df = utils.load_pickle_to_df(PICKLES_FOLDER + '/ratings.pickle').iloc[:,:4]
    objective_df = utils.load_pickle_to_df(PICKLES_FOLDER + '/objective.pickle').iloc[:,:4]

    # IMT-1 correlation
    for scale in ['valence','arousal','likeability','rewatch']:
        corrs = []
        p_vals = []
        for subj_id in SUBJECTS_IDS:
            obj = objective_df.xs(subj_id, level='subj_id').loc[:,scale].sort_index()
            subj = subjective_df.xs(subj_id, level='subj_id').loc[:,scale].sort_index()

            corr, p_val = pearsonr(obj, subj)


            # print(subj_id, obj, subj, corr)
            # print('------')


            if not isnan(corr):
                corrs.append(corr)
                p_vals.append(p_val)

        print('%s %.3f %.3f' % (scale, np.mean(corrs), np.std(corrs)))
