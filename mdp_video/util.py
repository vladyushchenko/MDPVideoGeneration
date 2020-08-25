import signal
from typing import Any, List, Generator, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils as vu
import docopt


class LoggerDict(dict):
    """
    Logger Dict.
    """

    def update(self, __m: Dict, **kwargs) -> None:  # type: ignore  # pylint: disable=W0613
        """
        Update dict.
        """
        for key, value in __m.items():
            if key in self.keys():
                self[key] += value
            else:
                self[key] = value


class Flags:
    """
    Args holder for models trainers.
    """

    def __init__(self, args: docopt.docopt, learning_rate: float) -> None:
        """
        Init call.

        :param args: program arguments
        :param learning_rate: learning rate
        """
        self.use_categories = args["--use_categories"]
        self.use_infogan = args["--use_infogan"]
        self.save_graph = args["--save_graph"]
        self.log_folder = args["<log_folder>"]
        self.optimizer = args["--optimizer"]
        self.learning_rate = learning_rate
        self.clamp_lower = float(args["--clamp_lower"])
        self.clamp_upper = float(args["--clamp_upper"])
        self.temporal_sigma = float(args["--temporal_sigma"])
        self.temporal_beta = float(args["--temporal_beta"])

    def __repr__(self) -> str:
        """
        Repr call.
        """
        contains = "    ".join("%s: %s \n" % item for item in vars(self).items())
        return "{} (\n {})".format(self.__class__.__name__, contains)


class SafeGuardKiller:
    """
    Class for safeguard.
    """

    kill_now = False

    def __init__(self) -> None:
        """
        Init call.
        """
        signal.signal(signal.SIGINT, self.exit_gracefully)  # type: ignore
        signal.signal(signal.SIGTERM, self.exit_gracefully)  # type: ignore

    def exit_gracefully(self, signum: int, frame: int) -> None:  # pylint: disable=W0613
        """
        Handle graceful exit.
        """
        self.kill_now = True


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert video to numpy.
    """
    generated = tensor.data.cpu().numpy()
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype("uint8")


def chunks(input_list: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(input_list) - chunk_size + 1):
        yield input_list[i : i + chunk_size]


def show_batch(
    batch: torch.Tensor, title: str = "", mode: str = "show", is_merged_disc: bool = False, max_videos: int = 1
) -> None:
    """
    Save/show current video batch fake or real.
    """
    normed = batch * 0.5 + 0.5
    is_video_batch = len(normed.size()) > 4

    if is_merged_disc:
        normed = normed[:max_videos, :, :3, :, :]
        normed = normed.permute(0, 2, 1, 3, 4)

    if is_video_batch:
        rows = []
        for b in normed:
            item = vu.make_grid(b.permute(1, 0, 2, 3).contiguous(), nrow=b.size(1)).cpu().detach().numpy()
            rows.append(item)
        im = np.concatenate(rows, axis=1)
    else:
        im = vu.make_grid(normed).cpu().numpy()

    im = im.transpose((1, 2, 0))

    if mode == "show":
        plt.figure()
        plt.title(title)
        plt.imshow(im)
        plt.show(block=True)
        plt.close()


class RealBatchSampler:
    """
    Wrapper for endless batch sampling.
    """

    def __init__(self, sampler: Any, args: docopt.docopt) -> None:
        self._batch_size: int = sampler.batch_size
        self._sampler = sampler
        self._enumerator: Optional[Any] = None
        self._cuda = args.cuda
        self._args = args

    def __iter__(self) -> Any:
        return self

    def __next__(self, *args, **kwargs) -> Tuple:  # type: ignore # pylint: disable=W0613
        """
        Sample provider call.
        """
        if self._enumerator is None:
            self._enumerator = enumerate(self._sampler)

        item = next(self._enumerator)
        try:
            assert len(item[1]) == 2
        except AssertionError:
            print(item)

        batch_idx, (batch, batch_cat) = next(self._enumerator)

        if self._cuda:
            batch = batch.cuda()
            batch_cat = batch_cat.cuda()

        if batch_idx == len(self._sampler) - 1:
            self._enumerator = enumerate(self._sampler)

        if self._args is not None:
            if self._args.add_noise_artifacts:
                noise = torch.empty_like(batch).normal_(0, self._args.noise_artifacts_std)
                batch_noise = batch + noise
                return batch_noise, batch
        return batch, batch_cat

    @property
    def batch_size(self) -> int:
        """
        Get batch size.
        """
        return self._batch_size

    def __len__(self) -> int:
        """
        Get length.
        """
        return len(self._sampler)
