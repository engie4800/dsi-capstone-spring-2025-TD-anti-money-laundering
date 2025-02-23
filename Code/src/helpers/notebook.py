from time import monotonic

from IPython import get_ipython


class CellTimer:
    def __init__(self):
        self.start_time = None

    def start(self, *args, **kwargs):
        self.start_time = monotonic()

    def stop(self, *args, **kwargs):
        try:
            delta = round(monotonic() - self.start_time, 2)
            print(f"\n⏱️ Execution time: {delta}s")
        except TypeError:
            # The `stop` will be called when the cell that
            # defines `CellTimer` is executed, but `start`
            # was never called, leading to a `TypeError` in
            # the subtraction. Skip it
            pass


def add_cell_timer() -> None:
    """
    Adds a cell timer to your notebook, printing the cell execution
    time on each cell run. Example usage:

        add_cell_timer()

    """
    timer = CellTimer()
    ipython = get_ipython()
    ipython.events.register("pre_run_cell", timer.start)
    ipython.events.register("post_run_cell", timer.stop)
