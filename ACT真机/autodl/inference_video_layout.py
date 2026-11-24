"""Shared layout for synchronized dual-camera and trajectory videos."""

from bisect import bisect_right
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SIZE = (1440, 1232)
BACKGROUND = "#171a1e"
IMAGE_BACKGROUND = "#20252a"
TEXT = "#e3e7eb"
MUTED = "#a5aeb7"
AXIS = "#79848e"
BASELINE = "#3b434c"
OBSERVED = "#91b8ce"
TARGET = "#607c9b"
CURSOR = "#677885"
CAMERA_BOXES = ((64, 80, 704, 560), (736, 80, 1376, 560))
PLOT_LEFT = 227
PLOT_RIGHT = 1375
TRACK_TOP = 620
TRACK_HEIGHT = 59
TRACK_STRIDE = 79


def tick_step(interval):
    magnitude = 10.0 ** math.floor(math.log10(interval))
    return next(factor * magnitude for factor in (1, 2, 2.5, 4, 5, 10)
                if factor * magnitude >= interval)


def axis_ticks(low, high, step):
    for index in range(math.ceil(low / step), math.floor(high / step) + 1):
        yield index * step


class VideoLayout:
    """Align two camera views and seven observed joint traces by sample index."""

    def __init__(self, title, subtitle, duration_s, joint_labels, observed, *,
                 max_observation_gap_s=None, targets=None):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Video duration must be positive and finite")
        self.duration_s = duration_s
        self.times, state_rows = zip(*observed)
        self.target_times, target_rows = zip(*targets) if targets else ((), ())
        self.segment_starts = [0]
        if max_observation_gap_s is not None:
            self.segment_starts.extend(
                index for index in range(1, len(self.times))
                if self.times[index] - self.times[index - 1] > max_observation_gap_s
            )
        self.tracks = []
        for index, (label, values) in enumerate(zip(joint_labels, zip(*state_rows, strict=True), strict=True)):
            top = TRACK_TOP + index * TRACK_STRIDE
            bottom = top + TRACK_HEIGHT
            requested = tuple(row[index] for row in target_rows)
            combined = values + requested
            minimum, maximum = min(combined), max(combined)
            step = tick_step(max(1, maximum - minimum) / 3)
            padding = max(0.5, step * 0.17)
            low, high = minimum - padding, maximum + padding
            degrees_to_pixels = TRACK_HEIGHT / (high - low)
            points = [(self._time_x(seconds), round(bottom - (value - low) * degrees_to_pixels))
                      for seconds, value in zip(self.times, values, strict=True)]
            commands = [(self._time_x(seconds), round(bottom - (value - low) * degrees_to_pixels))
                        for seconds, value in zip(self.target_times, requested, strict=True)]
            ticks = [(angle, round(bottom - (angle - low) * degrees_to_pixels))
                     for angle in axis_ticks(low, high, step)]
            self.tracks.append((label, top, bottom, ticks, points, commands))

        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        self.heading = ImageFont.truetype(str(fonts / "msyh.ttc"), 23)
        self.label = ImageFont.truetype(str(fonts / "msyh.ttc"), 19)
        self.small = ImageFont.truetype(str(fonts / "msyh.ttc"), 17)
        self.panel = ImageFont.truetype(str(fonts / "arialbd.ttf"), 24)
        self.base = self._build_background(title, subtitle)

    def _build_background(self, title, subtitle):
        frame = Image.new("RGB", SIZE, BACKGROUND)
        draw = ImageDraw.Draw(frame)
        draw.text((64, 26), title, font=self.heading, fill=TEXT, anchor="lm")
        draw.text((1376, 26), subtitle, font=self.small, fill=MUTED, anchor="rm")
        for box, letter, camera_name in zip(
            CAMERA_BOXES, "ab", ("顶部相机（D435i）", "腕部相机（D405）"),
        ):
            left, top, right, bottom = box
            draw.rectangle((left - 1, top - 1, right, bottom),
                           fill=IMAGE_BACKGROUND, outline=BASELINE)
            draw.text((left, 58), letter, font=self.panel, fill=TEXT, anchor="lm")
            draw.text((left + 28, 58), camera_name, font=self.label, fill=TEXT, anchor="lm")

        draw.text((64, 589), "c", font=self.panel, fill=TEXT, anchor="lm")
        if self.target_times:
            draw.text((92, 589), "七轴角度（°）", font=self.label, fill=TEXT, anchor="lm")
            for x, caption, color in ((265, "已发目标", TARGET), (416, "实测角度，缺采留白", OBSERVED)):
                draw.line((x, 589, x + 21, 589), fill=color, width=3)
                draw.text((x + 29, 589), caption, font=self.small, fill=MUTED, anchor="lm")
        else:
            draw.text((92, 589), "关节实测角度（°）", font=self.label, fill=TEXT, anchor="lm")
            draw.text((282, 589), "各轴独立刻度", font=self.small, fill=MUTED, anchor="lm")
        last_bottom = self.tracks[-1][2]
        for label, top, bottom, ticks, _, _ in self.tracks:
            draw.text((64, (top + bottom) // 2), label, font=self.label, fill=TEXT, anchor="lm")
            line_color = AXIS if bottom == last_bottom else BASELINE
            draw.line((PLOT_LEFT, bottom, PLOT_RIGHT, bottom), fill=line_color)
            draw.line((PLOT_LEFT, top, PLOT_LEFT, bottom), fill=AXIS)
            for angle, y in ticks:
                draw.line((PLOT_LEFT - 5, y, PLOT_LEFT, y), fill=AXIS)
                draw.text((PLOT_LEFT - 12, y), f"{angle:g}", font=self.small, fill=TEXT, anchor="rm")
        for seconds in axis_ticks(0, self.duration_s, tick_step(self.duration_s / 5)):
            x = self._time_x(seconds)
            draw.line((x, last_bottom, x, last_bottom + 5), fill=AXIS)
            draw.text((x, last_bottom + 9), f"{seconds:g}", font=self.small, fill=TEXT, anchor="mt")
        draw.text(((PLOT_LEFT + PLOT_RIGHT) // 2, 1209), "时间（s）",
                  font=self.label, fill=TEXT, anchor="mm")
        return frame

    def _time_x(self, seconds):
        return round(PLOT_LEFT + seconds / self.duration_s * (PLOT_RIGHT - PLOT_LEFT))

    def render(self, index: int, images: tuple[Image.Image | None, Image.Image | None]) -> Image.Image:
        return self._render(self.times[index], index + 1, images)

    def render_at(self, seconds, images):
        """Use the video clock while showing only recorded joint samples up to that time."""
        return self._render(seconds, bisect_right(self.times, seconds), images)

    def _render(self, seconds, sample_count, images):
        frame = self.base.copy()
        draw = ImageDraw.Draw(frame)
        draw.text((1376, 589), f"t = {seconds:05.2f} s",
                  font=self.small, fill=MUTED, anchor="rm")
        for (left, top, right, bottom), image in zip(CAMERA_BOXES, images):
            if image is None:
                draw.text(((left + right) // 2, (top + bottom) // 2), "暂无画面",
                          font=self.label, fill=MUTED, anchor="mm")
                continue
            picture = ImageOps.contain(image.convert("RGB"), (right - left, bottom - top),
                                       Image.Resampling.LANCZOS)
            frame.paste(picture, ((left + right - picture.width) // 2,
                                  (top + bottom - picture.height) // 2))

        x = self._time_x(seconds)
        if x > PLOT_LEFT:
            draw.line((x, self.tracks[0][1], x, self.tracks[-1][2] - 1), fill=CURSOR)
        starts = self.segment_starts[:bisect_right(self.segment_starts, sample_count - 1)]
        spans = tuple(zip(starts, starts[1:] + [sample_count]))
        command_count = bisect_right(self.target_times, seconds)
        for _, _, _, _, points, commands in self.tracks:
            if command_count > 1:
                draw.line(commands[:command_count], fill=TARGET, width=2)
            for start, stop in spans:
                if stop - start > 1:
                    draw.line(points[start:stop], fill=OBSERVED, width=2)
                else:
                    draw.point(points[start], fill=OBSERVED)
            if sample_count:
                point_x, y = points[sample_count - 1]
                draw.ellipse((point_x - 3, y - 3, point_x + 3, y + 3), fill=OBSERVED)
        return frame
