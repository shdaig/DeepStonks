import os
import shutil


def mkdir(dirname: str):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

        print(f"[OS] mkdir {dirname}/")


def rmdir(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

        print(f"[OS] rmdir {dirname}/")
