import argparse
import math

def set_custom_args(parser):
    parser.add_argument('--max_value', help="[CATS estimator] max value for continuous action range", required=False)
    parser.add_argument('--min_value', help="[CATS estimator] min value for continuous action range", required=False)
    parser.add_argument('--num_actions', help="[CATS estimator] number of actions used to discretize continuous range", required=False)
    parser.add_argument('--bandwidth', help="[CATS estimator] bandwidth (radius) of randomization around discrete actions in terms of continuous range ", required=False)
    return parser

def get_cats_transformer(args):
    if args.num_actions and args.max_value and args.min_value and args.bandwidth:
        return CatsTransformer(args.num_actions, args.bandwidth, args.max_value, args.min_value)
    else:
        return

class CatsTransformer:
    def __init__(self, num_actions, bandwidth, max_value, min_value):
        self.num_actions = int(num_actions)
        self.max_value = float(max_value)
        self.min_value = float(min_value)
        self.bandwidth = float(bandwidth)

        self.continuous_range = self.max_value - self.min_value
        self.unit_range = self.continuous_range / float(self.num_actions)
    
    def get_baseline1_prediction(self):
        return self.min_value + (self.unit_range / 2.0)
    
    def transform(self, data, pred_a):        
        logged_a = data['a']

        ctr = min((self.num_actions - 1), math.floor((pred_a - self.min_value) / self.unit_range))
        centre = self.min_value + ctr * self.unit_range + (self.unit_range / 2.0)

        if(math.isclose(centre, logged_a, abs_tol=self.bandwidth)):
            b = min(self.max_value, centre + self.bandwidth) - max(self.min_value, centre - self.bandwidth)
            data['pred_p'] = 1.0 / b
        else:
            data['pred_p'] = 0.0
        
        return data
