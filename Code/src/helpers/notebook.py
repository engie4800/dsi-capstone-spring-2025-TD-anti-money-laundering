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

    # Checks for existing events in the case of re-running the cell
    # that adds the cell timer in your notebook
    for f in list(ipython.events.callbacks.get("pre_run_cell", [])):
        breakpoint()
        if isinstance(f.__self__, CellTimer):
            # The timer has already been registered. This might cause
            # issues if you are changing the timer, as a new one won't
            # be registered
            return

    ipython.events.register("pre_run_cell", timer.start)
    ipython.events.register("post_run_cell", timer.stop)
