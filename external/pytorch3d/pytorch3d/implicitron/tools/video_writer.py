# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import warnings
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


matplotlib.use("Agg")


class VideoWriter:
    """
    A class for exporting videos.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        out_path: str = "/tmp/video.mp4",
        fps: int = 20,
        output_format: str = "visdom",
        rmdir_allowed: bool = False,
        **kwargs,
    ):
        """
        Args:
            cache_dir: A directory for storing the video frames. If `None`,
                a temporary directory will be used.
            ffmpeg_bin: The path to an `ffmpeg` executable.
            out_path: The path to the output video.
            fps: The speed of the generated video in frames-per-second.
            output_format: Format of the output video. Currently only `"visdom"`
                is supported.
            rmdir_allowed: If `True` delete and create `cache_dir` in case
                it is not empty.
        """
        self.rmdir_allowed = rmdir_allowed
        self.output_format = output_format
        self.fps = fps
        self.out_path = out_path
        self.cache_dir = cache_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.frames = []
        self.regexp = "frame_%08d.png"
        self.frame_num = 0

        if self.cache_dir is not None:
            self.tmp_dir = None
            if os.path.isdir(self.cache_dir):
                if rmdir_allowed:
                    shutil.rmtree(self.cache_dir)
                else:
                    warnings.warn(
                        f"Warning: cache directory not empty ({self.cache_dir})."
                    )
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.cache_dir = self.tmp_dir.name

    def write_frame(
        self,
        frame: Union[matplotlib.figure.Figure, np.ndarray, Image.Image, str],
        resize: Optional[Union[float, Tuple[int, int]]] = None,
    ):
        """
        Write a frame to the video.

        Args:
            frame: An object containing the frame image.
            resize: Either a floating defining the image rescaling factor
                or a 2-tuple defining the size of the output image.
        """

        # pyre-fixme[6]: For 1st param expected `Union[PathLike[str], str]` but got
        #  `Optional[str]`.
        outfile = os.path.join(self.cache_dir, self.regexp % self.frame_num)

        if isinstance(frame, matplotlib.figure.Figure):
            plt.savefig(outfile)
            im = Image.open(outfile)
        elif isinstance(frame, np.ndarray):
            if frame.dtype in (np.float64, np.float32, float):
                frame = (np.transpose(frame, (1, 2, 0)) * 255.0).astype(np.uint8)
            im = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            im = frame
        elif isinstance(frame, str):
            im = Image.open(frame).convert("RGB")
        else:
            raise ValueError("Cant convert type %s" % str(type(frame)))

        if im is not None:
            if resize is not None:
                if isinstance(resize, float):
                    resize = [int(resize * s) for s in im.size]
            else:
                resize = im.size
            # make sure size is divisible by 2
            resize = tuple([resize[i] + resize[i] % 2 for i in (0, 1)])
            im = im.resize(resize, Image.ANTIALIAS)
            im.save(outfile)

        self.frames.append(outfile)
        self.frame_num += 1

    def get_video(self, quiet: bool = True):
        """
        Generate the video from the written frames.

        Args:
            quiet: If `True`, suppresses logging messages.

        Returns:
            video_path: The path to the generated video.
        """

        # pyre-fixme[6]: For 1st param expected `Union[PathLike[str], str]` but got
        #  `Optional[str]`.
        regexp = os.path.join(self.cache_dir, self.regexp)

        if self.output_format == "visdom":  # works for ppt too
            ffmcmd_ = (
                "%s -r %d -i %s -vcodec h264 -f mp4 \
                       -y -crf 18 -b 2000k -pix_fmt yuv420p '%s'"
                % (self.ffmpeg_bin, self.fps, regexp, self.out_path)
            )
        else:
            raise ValueError("no such output type %s" % str(self.output_format))

        if quiet:
            ffmcmd_ += " > /dev/null 2>&1"
        else:
            print(ffmcmd_)
        os.system(ffmcmd_)

        return self.out_path

    def __del__(self):
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
