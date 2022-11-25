import os
from os.path import isfile, join

from datetime import timedelta as tdel
from datetime import datetime as dt


def shift_time_fwd(t, h=1, m=0):
    return t + tdel(hours=h, minutes=m)


def shift_time_bck(t, h=1, m=0):
    return t - tdel(hours=h, minutes=m)


def read_meals(photos_folder, shift_time_function=shift_time_fwd, hours=1, minutes=0):
    if not shift_time_function:
        # TODO log that there is no shift
        pass
    # get timestamp of meals
    meals_tsp = [
        (
            shift_time_function(
                dt.fromtimestamp(os.stat(photos_folder + f).st_mtime), hours, minutes
            ),
            f,
        )
        for f in os.listdir(photos_folder)
        if isfile(join(photos_folder, f))
    ]
    print([x[0] for x in meals_tsp])
