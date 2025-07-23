import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 28
    flags.num_threads = 190
    flags.load_model = True
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    flags.use_oracle_reward = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train(flags)
