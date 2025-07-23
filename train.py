import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 18
    flags.num_threads = 62
    flags.load_model = True
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    flags.use_oracle_reward = True
    flags.save_interval = 10
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train(flags)
