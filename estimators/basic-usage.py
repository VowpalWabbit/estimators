import argparse, os, gzip
from bandits import cressieread
from bandits import ips
from bandits import snips
from bandits import mle
from bandits import gaussian
from bandits import clopper_pearson
from bandits import cats_utils
from utils import ds_parse


def compute_estimates(log_fp, cats_transformer=None):
    # Init estimators
    online_ips = ips.Estimator()
    online_snips = snips.Estimator()
    online_mle = mle.Estimator()
    online_cressieread = cressieread.Estimator()

    baseline1_ips = ips.Estimator()
    baseline1_snips = snips.Estimator()
    baseline1_mle = mle.Estimator()
    baseline1_cressieread = cressieread.Estimator()

    baselineR_ips = ips.Estimator()
    baselineR_snips = snips.Estimator()
    baselineR_mle = mle.Estimator()
    baselineR_cressieread = cressieread.Estimator()

    baseline1_gaussian = gaussian.Interval()
    baseline1_clopper_pearson = clopper_pearson.Interval()

    baselineR_gaussian = gaussian.Interval()
    baselineR_clopper_pearson = clopper_pearson.Interval()

    print('Processing: {}'.format(log_fp))
    bytes_count = 0
    tot_bytes = os.path.getsize(log_fp)
    evts = 0
    for i,x in enumerate(gzip.open(log_fp, 'rb') if log_fp.endswith('.gz') else open(log_fp, 'rb')):
        # display progress
        bytes_count += len(x)
        if (i+1) % 10000 == 0:
            if log_fp.endswith('.gz'):
                ds_parse.update_progress(i+1)
            else:
                ds_parse.update_progress(bytes_count,tot_bytes)

        # parse dsjson file
        if x.startswith(b'{"_label_cost":') and x.strip().endswith(b'}'):
            data = ds_parse.json_cooked(x)

            if data['skipLearn']:
                continue

            r = 0 if data['cost'] == b'0' else -float(data['cost'])

            # Update estimators with tuple (p_log, r, p_pred)
            online_ips.add_example(data['p'], r, data['p'])
            baseline1_ips.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_ips.add_example(data['p'], r, 1/data['num_a'])

            online_snips.add_example(data['p'], r, data['p'])
            baseline1_snips.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_snips.add_example(data['p'], r, 1/data['num_a'])

            online_mle.add_example(data['p'], r, data['p'])
            baseline1_mle.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_mle.add_example(data['p'], r, 1/data['num_a'])

            online_cressieread.add_example(data['p'], r, data['p'])
            baseline1_cressieread.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_cressieread.add_example(data['p'], r, 1/data['num_a'])

            baseline1_gaussian.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baseline1_clopper_pearson.add_example(data['p'], r, 1 if data['a'] == 1 else 0)

            baselineR_gaussian.add_example(data['p'], r, 1/data['num_a'])
            baselineR_clopper_pearson.add_example(data['p'], r, 1/data['num_a'])

            evts += 1

        if x.startswith(b'{"_label_ca":') and x.strip().endswith(b'}'):
            data = ds_parse.json_cooked_continuous_actions(x)
            if cats_transformer is None:
                raise RuntimeError("Not all of the required arguments for running with continuous actions have been provided.")
            # passing logged action as predicted action to transformer
            data = cats_transformer.transform(data, data['a'])
            # passing baseline action as predicted action to transformer
            data_baseline1 = cats_transformer.transform(data, cats_transformer.get_baseline1_prediction())

            if data['skipLearn']:
                continue

            r = 0 if data['cost'] == b'0' else -float(data['cost'])

            # Update estimators with tuple (p_log, r, p_pred)
            online_ips.add_example(data['p'], r, data['p'])
            baseline1_ips.add_example(data['p'], r, data_baseline1['pred_p'])
            baselineR_ips.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)

            online_snips.add_example(data['p'], r, data['p'])
            baseline1_snips.add_example(data['p'], r, data_baseline1['pred_p'])
            baselineR_snips.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)

            online_mle.add_example(data['p'], r, data['p'])
            baseline1_mle.add_example(data['p'], r, data_baseline1['pred_p'])
            baselineR_mle.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)

            online_cressieread.add_example(data['p'], r, data['p'])
            baseline1_cressieread.add_example(data['p'], r, data_baseline1['pred_p'])
            baselineR_cressieread.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)

            baseline1_gaussian.add_example(data['p'], r, data_baseline1['pred_p'])
            baseline1_clopper_pearson.add_example(data['p'], r, data_baseline1['pred_p'])

            baselineR_gaussian.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)
            baselineR_clopper_pearson.add_example(data['p'], r, 1.0 / cats_transformer.continuous_range)
            
            evts += 1


    if log_fp.endswith('.gz'):
        len_text = ds_parse.update_progress(i+1)
    else:
        len_text = ds_parse.update_progress(bytes_count,tot_bytes)

    print('\nProcessed {} events out of {} lines'.format(evts,i+1))

    print('online_ips:',online_ips.get())

    print('baseline1_ips:', baseline1_ips.get())
    print('baseline1 gaussian ci:', baseline1_gaussian.get())
    print('baseline1 clopper pearson ci:', baseline1_clopper_pearson.get())

    print('baselineR_ips:',baselineR_ips.get())
    print('baselineR gaussian ci:', baselineR_gaussian.get())
    print('baselineR clopper pearson ci:', baselineR_clopper_pearson.get())


    print('online_snips:',online_snips.get())
    print('baseline1_snips:',baseline1_snips.get())
    print('baselineR_snips:',baselineR_snips.get())

    print('online_mle:',online_mle.get())
    print('baseline1_mle:',baseline1_mle.get())
    print('baselineR_mle:',baselineR_mle.get())

    print('online_cressieread:',online_cressieread.get())
    print('baseline1_cressieread:',baseline1_cressieread.get())
    print('baselineR_cressieread:',baselineR_cressieread.get())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--log_fp', help="data file path (.json or .json.gz format - each line is a dsjson)", required=True)
    parser = cats_utils.set_custom_args(parser)
    args = parser.parse_args()
    cats_transformer = cats_utils.get_cats_transformer(args)

    compute_estimates(args.log_fp, cats_transformer)
