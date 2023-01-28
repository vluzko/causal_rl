import datetime
import os
import time

import torch
import wandb


class Tracker:
    def add_histogram(self, tag, data, i):
        raise NotImplementedError

    def add_dictionary(self, dict):
        raise NotImplementedError

    def add_scalar(self, tag, value, i):
        raise NotImplementedError

    def add_image(self, tag, value, i):
        raise NotImplementedError

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        raise NotImplementedError

    def set_summary(self, key, value):
        raise NotImplementedError


class WandBTracker(Tracker):
    def __init__(self, name=None, config=None):
        wandb.init(entity="nri_mbrl", project=name, config=config)
        if config is not None:
            if "name" in config:
                wandb.run.name = config["name"]  # type: ignore

    def add_histogram(self, tag, data, i):
        if type(data) == torch.Tensor:
            data = data.cpu().detach()
        wandb.log({tag: wandb.Histogram(data)}, step=i)

    def add_dictionary(self, dict):
        wandb.log(dict)

    def add_scalar(self, tag, value, i):
        wandb.log({tag: value}, step=i)

    def add_image(self, tag, value, i):
        wandb.log({tag: [wandb.Image(value, caption="Label")]}, step=i)

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  # noqa
            self.last_time = time.time()
            self.add_scalar("timings/iterations-per-sec", 1 / dt, i)
            self.add_scalar("timings/samples-per-sec", batch_size / dt, i)
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        wandb.run.summary[key] = value  # type: ignore


class ConsoleTracker(Tracker):
    def __init__(self, name=None, config=None):
        pass

    def add_histogram(self, tag, data, i):
        pass

    def add_dictionary(self, dict):
        pass

    def add_scalar(self, tag, value, i):
        print(f"{i}  {tag}: {value}")

    def add_image(self, tag, value, i):
        pass

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  # noqa
            self.last_time = time.time()
            print(f"{i}  iterations-per-sec: {1/dt}")
            print(f"{i}  samples-per-sec: {batch_size/dt}")
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        print(f"{key}: {value}")


class TensorboardTracker(Tracker):
    pass
