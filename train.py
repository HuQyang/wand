import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 4
    flags.num_threads = 64
    flags.load_model = True
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    flags.use_oracle_reward = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
