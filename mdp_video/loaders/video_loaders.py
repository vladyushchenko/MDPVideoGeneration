import os
import sys
import warnings
from typing import Callable, Tuple, Any, Optional

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import transforms
import docopt
import chainer


from mdp_video.loaders.ucf_loader import SplittedVideoDataset
from mdp_video.util import to_numpy, RealBatchSampler


class Sampler:
    """
    Sampler class.
    """

    def __init__(self, generator: Any, transform: Callable, args: docopt.docopt) -> None:
        """
        Init call.

        :param generator: generator network
        :param transform: transformation
        :param args: program args
        """
        self._generator = generator
        self._transforms = transform
        self._args = args
        self._counter = 0

    def __iter__(self) -> Any:
        """
        Get iterator.
        """
        return self

    def __next__(self) -> Tuple:
        """
        Get next element.

        :return: element from dataset
        :raises StopIteration: in case iterator is over.
        """
        if (self._counter + 1) * self._args.batch_size <= self._args.calc_iter:
            seed = None
            if self._args.seed >= 0:
                seed = self._counter
            x, _ = self._generator.sample_videos(
                num_samples=self._args.batch_size, video_len=self._args.video_length, seed=seed
            )
            if x.shape[-2:] != (self._args.image_size, self._args.image_size):
                print("Interpolation images to size: {}x{}".format(self._args.image_size, self._args.image_size))
                x = self._interpolate_images(x, mode="cubic")
            self._counter += 1
            return x, None

        self._counter = 0
        raise StopIteration

    def __len__(self) -> int:
        """
        Get length of dataset.
        """
        return int(self._args.calc_iter // self._args.batch_size)

    def _interpolate_images(self, batch_images: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Interpolate images for loader.
        """
        if mode == "linear":
            images = interpolate(
                batch_images,
                mode="trilinear",
                size=(self._args.video_length, self._args.image_size, self._args.image_size),
            )
        elif mode == "cubic":
            splitted_images = torch.split(batch_images, 1, dim=0)

            return_video = []
            for images in splitted_images:
                sequence = torch.split(images, 1, dim=2)

                raw_images = []
                for image in sequence:
                    image = image.squeeze()
                    # we lose here some accuracy compared to linear due to float32 -> uint8 -> float32
                    image = to_numpy(image).transpose(1, 2, 0)
                    image = Image.fromarray(image)
                    image = self._transforms(image)

                    raw_images.append(image)
                video = torch.stack(raw_images, dim=1)
                return_video.append(video)

            images = torch.stack(return_video)
        else:
            warnings.warn("Batch was not resized. This may lead to crash!")
            images = batch_images

        return images


class ArtifactSampler:
    """
    Wrapper for endless batch sampling.
    """

    def __init__(self, sampler: Any, args: docopt.docopt, cuda: bool = True) -> None:
        """
        Init call.

        :param sampler:
        :param args:
        :param cuda:
        """
        self._batch_size = sampler.batch_size
        self._sampler = sampler
        self._enumerator: Optional[Any] = None
        self._cuda = cuda
        self._args = args

    def __iter__(self) -> Any:
        return self

    def __next__(self, *args, **kwargs) -> Tuple:  # type: ignore  # pylint: disable=W0613
        """
        Sample provider call.
        """
        if self._enumerator is None:
            self._enumerator = enumerate(self._sampler)

        batch_idx, (batch, batch_cat) = next(self._enumerator)

        if batch_idx == len(self._sampler) - 1:
            self._enumerator = enumerate(self._sampler)

        if self._args.artifact == "looping_forward":
            repeats = self._args.n_frames // self._args.non_artifact_length
            assert repeats * self._args.non_artifact_length == self._args.n_frames
            batch = batch[:, :, : self._args.non_artifact_length, :, :].repeat(1, 1, repeats, 1, 1)
        elif self._args.artifact == "looping_backward":
            repeats = self._args.n_frames // self._args.non_artifact_length
            assert repeats * self._args.non_artifact_length == self._args.n_frames
            looped_clip_forward = batch[:, :, : self._args.non_artifact_length // 2, :, :]
            looped_clip_backward = looped_clip_forward.cpu().numpy()
            looped_clip_backward = np.ascontiguousarray(looped_clip_backward[:, :, ::-1, :, :])
            looped_clip_backward = torch.from_numpy(looped_clip_backward)
            looped_clip = torch.cat((looped_clip_forward, looped_clip_backward), dim=2)
            batch = looped_clip.repeat(1, 1, repeats, 1, 1)
        elif self._args.artifact == "freezing":
            assert self._args.non_artifact_length >= 1
            repeats = self._args.n_frames - self._args.non_artifact_length
            last_image = batch[:, :, (self._args.non_artifact_length - 1) : self._args.non_artifact_length, :, :]
            freezed = last_image.repeat(1, 1, repeats, 1, 1)
            batch = torch.cat((batch[:, :, : self._args.non_artifact_length, :, :], freezed), dim=2)
        else:
            raise RuntimeError

        if self._cuda:
            batch = batch.cuda()
            batch_cat = batch_cat.cuda()

        if self._args.add_noise_artifacts:
            noise = torch.empty_like(batch).normal_(0, self._args.noise_artifacts_std)
            batch = batch + noise

        return batch, batch_cat

    @property
    def batch_size(self) -> int:
        """
        Return batch size.

        :return: batch size
        """
        return int(self._batch_size)

    def __len__(self) -> int:
        """
        Return length of data.

        :return: data len
        """
        return int(self._args.calc_iter // self._args.batch_size)


class Loader:
    """
    Loader class.
    """

    def __init__(self, args: docopt.docopt) -> None:
        """
        Init call.

        :param args: program args
        """
        self._args = args
        self._transform = transforms.Compose(
            [
                # transforms.Resize(self._args.image_size, interpolation=Image.CUBIC),
                transforms.CenterCrop(self._args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _check_arguments(self) -> None:
        """
        Check validity of args.

        :raises RuntimeError: in case of bad args
        """

        attributes_needed = [
            "location",
            "mode",
            "image_size",
            "batch_size",
            "cuda",
            "calc_iter",
            "num_workers",
            "video_length",
            "seed",
            "every_nth",
        ]
        attr_in_args = [attr for attr in attributes_needed if attr not in self._args.__dict__.keys()]
        if attr_in_args:
            raise RuntimeError('Arguments "{}" not found'.format(attr_in_args))

    def __call__(self, *argv, **kwargs) -> Any:  # type: ignore
        """
        Create torch loader of dataset or using generator.
        """
        self._check_arguments()
        if self._args.seed >= 0:
            np.random.seed(self._args.seed)
            torch.cuda.manual_seed(self._args.seed)
            torch.manual_seed(self._args.seed)

        if not os.path.exists(self._args.location):
            raise RuntimeError("Invalid dataset location. {} not a folder".format(self._args.location))

        # pylint: disable = R1705
        if self._args.mode == "image":
            print("Using dataset {}".format(self._args.location))

            dataset = SplittedVideoDataset(
                root_path=self._args.location,
                spatial_transform=self._transform,
                sample_duration=self._args.video_length,
                n_samples=self._args.every_nth,
            )
            sample_provider = torch.utils.data.DataLoader(
                dataset,
                batch_size=self._args.batch_size,
                shuffle=True,
                num_workers=self._args.num_workers,
                drop_last=True,
            )
            return RealBatchSampler(sample_provider, self._args)

        elif self._args.mode == "artifact":
            print("Using dataset {}".format(self._args.location))

            dataset = SplittedVideoDataset(
                root_path=self._args.location,
                spatial_transform=self._transform,
                sample_duration=self._args.video_length,
                n_samples=self._args.every_nth,
            )
            sample_provider = torch.utils.data.DataLoader(
                dataset,
                batch_size=self._args.batch_size,
                shuffle=True,
                num_workers=self._args.num_workers,
                drop_last=True,
            )
            sample_provider = ArtifactSampler(sample_provider, self._args)
            return sample_provider

        elif self._args.mode == "generator":
            print("Using generator {}".format(self._args.model))

            generator = torch.load(self._args.model)
            generator.eval()
            if self._args.cuda:
                generator.cuda()

            return Sampler(generator, self._transform, self._args)

        elif self._args.mode == "tgan_generator":

            sys.path.insert(0, "<path to tgan project>")  # isort:skip
            # pylint: disable = C0415
            from infer import get_models
            from infer import make_video

            gpu = 0
            n_iter = 100000

            chainer.cuda.get_device(gpu).use()
            chainer.cuda.cupy.random.seed(self._args.seed)
            fsgen, vgen, _ = get_models(self._args.location, n_iter)

            if gpu >= 0:
                fsgen.to_gpu()
                vgen.to_gpu()

            class GenTGAN:
                def __iter__(self) -> Any:
                    return self

                @staticmethod
                def __next__() -> Tuple:
                    y, _, _ = make_video(fsgen, vgen, n=1)
                    y[y < -1] = -1
                    y[y > 1] = 1
                    y = (y + 1) / 2 * 255
                    return y, None

            return GenTGAN()
        else:
            raise RuntimeError(f"Mode {self._args.mode} is not understood.")
