import numpy as np
from IPython.display import clear_output


def rnd_split(ndarray, size):
    count = (len(ndarray)//size) + 1
    return np.array_split(ndarray, count)

def print_progress(progress, bar_length=20):
    clear_output(wait=True)
    block = int(round(bar_length * progress))
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

class Progress:
    def __init__(self, steps):
        self.current = 0
        self.steps = steps

    def reset(self):
        self.current = 0
        self.print()

    def bump(self, i=1):
        assert(self.current < self.steps, "Did you forget to reset the progress bar?")
        self.current += i
        self.print()

    def print(self):
        print_progress(self.current / self.steps)