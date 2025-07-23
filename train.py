import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 4
    flags.num_threads = 62
    flags.load_model = True
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    flags.use_oracle_reward = True
    flags.save_interval = 120
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
