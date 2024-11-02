from training import train_runner


def run(platform):
    if platform.opt.pre_train is True:
        train_runner.run(platform, mode='pre')
