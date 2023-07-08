from detectron2.utils.events import CommonMetricPrinter, EventWriter, get_event_storage
import datetime
import time
import torch

class CustomMetricPrinter(CommonMetricPrinter):
    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None


        losses_text ="  ".join(
                        [
                            "{}: {:.4g}".format(k, v.median(self._window_size))
                            for k, v in storage.histories().items()
                            if ("loss" in k or "Loss" in k) and ("val" not in k)
                        ]
                    )
        acc_text =  "  ".join(
                        [
                            "{}: {:.4g}".format(k, v.median(self._window_size))
                            for k, v in storage.histories().items()
                            if ("acc" in k or "Acc" in k) and ("val" not in k)
                        ]
                    )

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}  {losses} {accuracies} {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses=losses_text,
                accuracies=acc_text,
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )

class CalcWriter(EventWriter):
    """
    Write scalars to a csv file.

    It saves headers on top of file.
    It saves scalars on next free line.

    Examples of such a file:
    :: "data_time"; "iteration"; "loss"; "loss_box_reg"; ...

    """
    def __init__(self, csv_file, epoch_size, separator: str = ";", window_size: int = 20):
        self.csv_file = csv_file
        self.epoch_size = epoch_size
        self.separator = separator
        self._headers_set = False
        self._window_size = window_size
        self.header_names = []
        self.prev_values = {}

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        headers = ["epoch", "iter"]
        values = [iteration / self.epoch_size, iteration]
        pos_metric = len(headers)
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if self._headers_set:
                # check order of headers saved is the same as keys in header_names
                if pos_metric>=len(self.header_names):
                    self._headers_set = False
                elif k!=self.header_names[pos_metric]:
                    self._headers_set = False
            headers.append(k)

            # print only new values - not print evaluation repeated values
            if not self._headers_set or k not in self.prev_values.keys():
                self.prev_values[k] = v
                values.append(v)
            elif v == self.prev_values[k]:
                values.append('')
            else:
                self.prev_values[k] = v
                values.append(v)

            pos_metric += 1

        if not self._headers_set:
            self.write_headers(headers)
        self.write_data_log(values)

    def write_headers(self, headers):
        with open(self.csv_file, 'a') as fh:
            for head in headers:
                fh.write(head + self.separator)
            fh.write('\n')
        self._headers_set = True
        self.header_names = headers

    def write_data_log(self, data_array):
        with open(self.csv_file, 'a') as fh:
            for data_item in data_array:
                fh.write(str(data_item)+ self.separator)
            fh.write('\n')

