import argparse, os, gzip
import ips_snips
import mle
import ds_parse


def compute_estimates(log_fp):
    # Init estimators
    online = ips_snips.Estimator()
    baseline1 = ips_snips.Estimator()
    baselineR = ips_snips.Estimator()
    online_mle = mle.Estimator()
    baseline1_mle = mle.Estimator()
    baselineR_mle = mle.Estimator()

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
            online.add_example(data['p'], r, data['p'])
            baseline1.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR.add_example(data['p'], r, 1/data['num_a'])

            online_mle.add_example(data['p'], r, data['p'])
            baseline1_mle.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_mle.add_example(data['p'], r, 1/data['num_a'])

            evts += 1

    if log_fp.endswith('.gz'):
        len_text = ds_parse.update_progress(i+1)
    else:
        len_text = ds_parse.update_progress(bytes_count,tot_bytes)

    print('\nProcessed {} events out of {} lines'.format(evts,i+1))
                        
    print('online_ips:',online.get_estimate('ips'))
    print('baseline1_ips:',baseline1.get_estimate('ips'))
    print('baselineR_ips:',baselineR.get_estimate('ips'))

    print('online_snips:',online.get_estimate('snips'))
    print('baseline1_snips:',baseline1.get_estimate('snips'))
    print('baselineR_snips:',baselineR.get_estimate('snips'))

    print('online_mle:',online_mle.get_estimate())
    print('baseline1_mle:',baseline1_mle.get_estimate())
    print('baselineR_mle:',baselineR_mle.get_estimate())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--log_fp', help="data file path (.json or .json.gz format - each line is a dsjson)", required=True)

    args = parser.parse_args()

    compute_estimates(args.log_fp)