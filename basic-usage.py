import argparse, os, gzip
import cressieread
import ips_snips
import mle
import ds_parse
import dr
import policy
from vowpalwabbit import pyvw
import os

def compute_estimates(log_fp, dr_mode, pol):
    """
        Using a logging policy, computes estimates of expected rewards for different target policies as well as confidence intervals around such estimates, and reports the results

        Parameters
        ----------
        log_fp : file(json)
            Logging policy
        dr_mode: string
            The mode of the doubly robust estimator, can be {batch, online}
        pol : string
            target policy to evaluate
            
        Returns
        -------
        None
        """
    
    # Init estimators
    online = ips_snips.Estimator()
    baseline1 = ips_snips.Estimator()
    baselineR = ips_snips.Estimator()
    online_mle = mle.Estimator()
    baseline1_mle = mle.Estimator()
    baselineR_mle = mle.Estimator()
    online_cressieread = cressieread.Estimator()
    baseline1_cressieread = cressieread.Estimator()
    baselineR_cressieread = cressieread.Estimator()

    dre_batch = dr.Estimator()
    baseline1_dre_batch = dr.Estimator()
    baselineR_dre_batch = dr.Estimator()

    dre_online = dr.Estimator()
    baseline1_dre_online = dr.Estimator()
    baselineR_dre_online = dr.Estimator()

    reward_estimator_model = pyvw.vw(quiet=True)

    if(pol=='constant'):
        policy_pi = policy.Constant(10, 5)
    elif(pol=='uniformrandom'):
        policy_pi = policy.UniformRandom(10)

    lines_count = sum(1 for line in open(log_fp))
    train_ratio = 0.5
    train_size = int(lines_count * train_ratio)

    if(dr_mode=='batch'):
        print("Batch train size: ", train_size)

    load_pretrained_model = False
    train_data_str = ""
    train_data_file = open("xar.dat","w+")
    train_data_batch = []

    print('\n Computing estimates... Processing: {}'.format(log_fp))
    bytes_count = 0
    tot_bytes = os.path.getsize(log_fp)
    evts = 0
    idxx = 0

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

            data = ds_parse.ds_json_parse(x)

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

            online_cressieread.add_example(data['p'], r, data['p'])
            baseline1_cressieread.add_example(data['p'], r, 1 if data['a'] == 1 else 0)
            baselineR_cressieread.add_example(data['p'], r, 1/data['num_a'])
            
            if(dr_mode == 'online'):
                #construct training example (x, a, r)
                train_example = str(data['r']) + " |ANamespace Action:" + str(data['a']) + " |SharedNamespace c_user:" + str(data['c_user']) + " c_time_of_day:" + str(data['c_time_of_day'])
                
                #At the beginning, you don't have a model to predict with, so you can only learn a reward estimator
                if(idxx==0):
                    reward_estimator_model = dre_online.reward_estimator(train_example, reward_estimator_model, dr_mode, load_pretrained_model)
                    
                #Now you have a reward estimator to predict with
                elif(idxx>1):
                    
                    x = " |SharedNamespace c_user:" + str(data['c_user']) + " c_time_of_day:" + str(data['c_time_of_day'])
                    
                    #compute estimate
                    dre_online.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], data['p'], policy_pi.get_action_probas(0, 0), reward_estimator_model)
                    baseline1_dre_online.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], 1 if data['a'] == 1 else 0, policy_pi.get_action_probas(0, 0), reward_estimator_model)
                    baselineR_dre_online.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], 1/data['num_a'], policy_pi.get_action_probas(0, 0), reward_estimator_model)

                    #Update reward estimator
                    reward_estimator_model = dre_online.reward_estimator(train_example, reward_estimator_model, dr_mode, load_pretrained_model)

            elif(dr_mode == 'batch'):
                
                #accumulate training examples
                if(idxx < train_size):
                    train_example = str(data['r']) + " |ANamespace Action:" + str(data['a']) + " |SharedNamespace c_user:" + str(data['c_user']) + " c_time_of_day:" + str(data['c_time_of_day'])
                    
                    train_data_batch.append(train_example)
                    train_data_file.write(train_example + "\r\n")
                
                #use batch training examples to train a reward estimator
                else:
                    if(idxx == train_size):
                        train_data_file.close()
                        reward_estimator_model = dre_batch.reward_estimator(train_data_batch, reward_estimator_model, dr_mode, load_pretrained_model)
                    
                    #use reward estimator to compute estimate
                    x = " |SharedNamespace c_user:" + str(data['c_user']) + " c_time_of_day:" + str(data['c_time_of_day'])
                    dre_batch.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], data['p'], policy_pi.get_action_probas(0, 0), reward_estimator_model)
                    baseline1_dre_batch.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], 1 if data['a'] == 1 else 0, policy_pi.get_action_probas(0, 0), reward_estimator_model)
                    baselineR_dre_batch.add_example(x, data['a'], data['a_vec'], data['p'], data['r'], 1/data['num_a'], policy_pi.get_action_probas(0, 0), reward_estimator_model)

            evts += 1
            idxx += 1

    if log_fp.endswith('.gz'):
        len_text = ds_parse.update_progress(i+1)
    else:
        len_text = ds_parse.update_progress(bytes_count,tot_bytes)

    
    print('\nProcessed {} events out of {} lines'.format(evts,i+1))

    print('\nonline_ips:',online.get_estimate('ips'))

    print('baseline1_ips:', baseline1.get_estimate('ips'))
    print('baseline1 gaussian ci:', baseline1.get_interval('gaussian'))
    print('baseline1 clopper pearson ci:', baseline1.get_interval('clopper-pearson'))

    print('\nbaselineR_ips:',baselineR.get_estimate('ips'))
    print('baselineR gaussian ci:', baselineR.get_interval('gaussian'))
    print('baselineR clopper pearson ci:', baselineR.get_interval('clopper-pearson'))

    print('\nonline_snips:',online.get_estimate('snips'))
    print('baseline1_snips:',baseline1.get_estimate('snips'))
    print('baselineR_snips:',baselineR.get_estimate('snips'))

    print('\nonline_mle:',online_mle.get_estimate())
    print('baseline1_mle:',baseline1_mle.get_estimate())
    print('baselineR_mle:',baselineR_mle.get_estimate())

    print('\nonline_cressieread:',online_cressieread.get_estimate())
    print('baseline1_cressieread:',baseline1_cressieread.get_estimate())
    print('baselineR_cressieread:',baselineR_cressieread.get_estimate())

    if(dr_mode=='online'):
        print('\ndre_online:', dre_online.get_estimate())
        print('baseline1_dre_online:', baseline1_dre_online.get_estimate())
        print('baselineR_dre_online:', baselineR_dre_online.get_estimate())
    elif(dr_mode=='batch'):
        print('\ndre_batch:', dre_batch.get_estimate())
        print('baseline1_dre_batch:', baseline1_dre_batch.get_estimate())
        print('baselineR_dre_batch:', baselineR_dre_batch.get_estimate())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--log_fp', help="data file path (.json or .json.gz format - each line is a dsjson)", required=True)
    parser.add_argument('-m','--mode', help="Doubly Robust estimator mode", required=True)
    parser.add_argument('-p','--policy', help="Policy to evaluate one of {constant, uniformrandom}", required=True)
    
    args = parser.parse_args()

    compute_estimates(args.log_fp, args.mode, args.policy)
