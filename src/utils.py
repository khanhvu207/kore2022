import time

# ------------------------------------------------------------------------
# Modified from detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d %H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        self.start = time.time() * 1000

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def reset(self):
        self.start = time.time() * 1000

    def get_time_since_start(self, format=None):
        return self.get_time_hhmmss(self.start, format)

    def unix_time_since_start(self, in_seconds=True):
        gap = time.time() * 1000 - self.start

        if in_seconds:
            gap = gap // 1000

        # Prevent 0 division errors
        if gap == 0:
            gap = 1
        return gap

    def get_time_hhmmss(self, start=None, end=None, gap=None, format=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None and gap is None:

            if format is None:
                format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(format)

        if end is None:
            end = time.time() * 1000
        if gap is None:
            gap = end - start

        s, ms = divmod(gap, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        if format is None:
            format = self.DEFAULT_TIME_FORMAT

        items = [ms, s, m, h]
        assert len(items) == len(format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = format[idx] % item + " " + time_str

        # Means no more time is left.
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()
    
def profile(profiler, name, debug=True):
    if debug:
        print(name + ": " + profiler.get_time_since_start())
    profiler.reset()


class SaveOutputHook:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


fleet_w2i = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "N": 10,
    "E": 11,
    "S": 12,
    "W": 13,
    "C": 14,
    "CLS": 15,
    "EOS": 16,
    "PAD": 17,
}
shipyard_w2i = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "N": 10,
    "E": 11,
    "S": 12,
    "W": 13,
    "C": 14,
    "CLS": 15,
    "EOS": 16,
    "PAD": 17,
}
fleet_dir = {
    0: "N",
    1: "E",
    2: "S",
    3: "W"
}
action_encoder = {
    "IDLE": 0,
    "SPAWN": 1,
    "LAUNCH": 2
}
shipyard_i2w = {v: k for k, v in shipyard_w2i.items()}