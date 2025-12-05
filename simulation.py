from typing import Any
import numpy as np
import astropy.units as u
from dataclasses import dataclass
from numpy.typing import NDArray
from poppy import ArrayOpticalElement, OpticalSystem
from typing import cast, Optional
from enum import Enum

class StretchType(str, Enum):
    LOG = "log"
    AUTO = "auto"

def meters_to_pixels(x, y, pixels, pixel_scale):
    cx = pixels / 2
    cy = pixels / 2
    x_pix = cx + x / pixel_scale
    y_pix = cy - y / pixel_scale

    return x_pix, y_pix

def draw_line_segment_on_image(img, x0, y0, x1, y1, width, pixel_scale):
    pixels_y, pixels_x = img.shape

    x0, y0 = meters_to_pixels(x0, y0, pixels_x, pixel_scale)
    x1, y1 = meters_to_pixels(x1, y1, pixels_x, pixel_scale)
    width = width / pixel_scale

    x_min = max(int(min(x0, x1) - width // 2), 0)
    x_max = min(int(max(x0, x1) + width // 2) + 1, pixels_x)
    y_min = max(int(min(y0, y1) - width // 2), 0)
    y_max = min(int(max(y0, y1) + width // 2) + 1, pixels_y)
    
    meshgrid_x, meshgrid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    
    dx = x1 - x0
    dy = y1 - y0
    line_length = np.hypot(dx, dy)
    
    if line_length == 0:
        dist = np.hypot(meshgrid_x - x0, meshgrid_y - y0)
        mask = dist <= width / 2
    else:
        dist = np.abs(dy * meshgrid_x - dx * meshgrid_y + x1*y0 - y1*x0) / line_length
        t = ((meshgrid_x - x0) * dx + (meshgrid_y - y0) * dy) / (line_length**2)
        mask = (dist <= width / 2) & (t >= 0) & (t <= 1)
    
    img[y_min:y_max, x_min:x_max][mask] = 0.0

@dataclass
class Pupil:
    radius: u.Quantity
    pixels: int
    xx: NDArray[Any]
    yy: NDArray[Any]
    pupil: NDArray[Any]
    secondaryRadius: u.Quantity

    def __init__(self, radius: u.Quantity, pixels: int):
        x = np.linspace(-radius.value, radius.value, pixels) * u.m
        self.radius = radius
        self.pixels = pixels
        self.xx, self.yy = np.meshgrid(x, x)
        self.pupil = ((self.xx**2 + self.yy**2) <= radius**2).astype(float)
        self.secondaryRadius = 0 * u.m

    def get_pupil_pixel_scale(self):
        return cast(u.Quantity, 2*self.radius/(self.pixels * u.pixel))

    def add_secondary(self, radius):
        self.pupil *= ((self.xx**2 + self.yy**2) >= radius**2).astype(float)
        self.secondaryRadius = radius

    def add_offset_supports(
            self,
            width: u.Quantity,
            offset: u.Quantity,
            misalignment = 0.0,
            strut_length: u.Quantity = 0.0 * u.m,
            strut_width: u.Quantity = 0.0 * u.m
            ):
        pixel_scale = self.get_pupil_pixel_scale()
        l = self.radius.value
        o = offset.value
        print(f"Length: {l}, Offset: {o}, Misalignment: {misalignment}")
        vanes = [
                (0, o, l, o + misalignment, 0),
                (0, -o, -l, -o, 0),
                (o, 0, o, l, 0),
                (-o, 0, -o, -l, 0),
        ]

        for (x0, y0, x1, y1, thick_error) in vanes:
            dx = x1 - x0
            dy = y1 - y0
            u = np.array([x0, y0], dtype=float)
            v = np.array([x1, y1], dtype=float)
            dv = np.array([dx, dy], dtype=float)
            dv /= np.hypot(dx, dy)
            strut1 = dv * strut_length.value
            strut2 = dv * (strut_length.value * 0.5 + self.secondaryRadius.value)
            [sx1, sy1] = v - strut1
            [sx2, sy2] = u + strut2
            draw_line_segment_on_image(self.pupil, x0, y0, x1, y1, width.value * (1 + thick_error), pixel_scale.value)
            draw_line_segment_on_image(self.pupil, x1, y1, sx1, sy1, strut_width.value, pixel_scale.value)
            draw_line_segment_on_image(self.pupil, x0, y0, sx2, sy2, strut_width.value, pixel_scale.value)

    def to_optical_system(self, arcsec_per_pixel, fov_pixels=800):
        pixelscale = 2*self.radius/(self.pixels * u.pixel)
        array_pupil = ArrayOpticalElement(
                transmission=self.pupil,
                pixelscale=pixelscale,
        )

        osys = OpticalSystem(pupil_diameter=2*self.radius)
        osys.add_pupil(array_pupil)

        detector_pixelscale = arcsec_per_pixel * u.arcsec / u.pixel
        osys.add_detector(pixelscale=detector_pixelscale, fov_pixels=fov_pixels)

        return osys
    
    def display_on(self, axis):
        axis.axis('off')
        axis.imshow(
            self.pupil,
            origin='lower',
            cmap='gray',
            extent=[-self.radius.value, self.radius.value,
                    -self.radius.value, self.radius.value]
        )
        axis.set_title("Spider Shape")

red_high = 620e-9
red_low = 700e-9
green_high = 480e-9
green_low = 570e-9
blue_high = 420e-9
blue_low = 510e-9

def spectrum(low, high, dwave=10e-9):
    return np.arange(low, high, dwave)

rgb = [
    spectrum(red_high, red_low),
    spectrum(green_high, blue_low, 10*1e-9),
    spectrum(blue_high, blue_low, 10*1e-9),
]

def auto_stretch(
    img,
    low_pct=0.25,
    high_pct=99.9,
    use_asinh=True,
    asinh_a=10.0
):
    arr = np.asarray(img, float)
    arr = np.clip(arr, 0, np.inf)

    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        hi = lo + 1e-12

    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)

    if use_asinh:
        a = float(asinh_a)
        norm = np.arcsinh(a * norm) / np.arcsinh(a)

    return norm

def simulate_psf(
    colors,
    osys,
    stretch: StretchType = StretchType.AUTO
):
    """
    Returns an (ny, nx, nch) array with values in [0,1].

    - stretch == StretchType.LOG:
        compute log10(psf + eps) per-channel, then normalize using the
        global min/max across all channels (preserves color balance).
    - stretch == StretchType.AUTO:
        apply auto_stretch() per-channel (percentile + asinh).
    """
    eps = 1e-12
    channels = []

    if stretch == StretchType.LOG:
        # compute log images for every channel first
        log_channels = []
        for wavelengths in colors:
            source = {
                "wavelengths": wavelengths,
                "weights": np.ones_like(wavelengths)
            }
            psf = osys.calc_psf(source=source)
            psf_data = psf[0].data.astype(float)
            # ensure non-negative then take log10
            psf_data = np.clip(psf_data, 0.0, np.inf)
            log_img = np.log10(psf_data + eps)
            log_channels.append(log_img)

        log_stack = np.stack(log_channels, axis=0)  # (nchan, ny, nx)
        # global min/max across all channels (preserves color balance)
        min_val = log_stack.min()
        max_val = log_stack.max()
        # avoid degenerate range
        if max_val <= min_val + 1e-15:
            max_val = min_val + 1.0
        norm = (log_stack - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0.0, 1.0)
        # move channel axis to last -> (ny, nx, nchan)
        return np.moveaxis(norm, 0, -1)

    elif stretch == StretchType.AUTO:
        for wavelengths in colors:
            source = {
                "wavelengths": wavelengths,
                "weights": np.ones_like(wavelengths)
            }

            psf = osys.calc_psf(source=source)
            psf_data = psf[0].data.astype(float)

            # use the auto_stretch you already have (percentile + asinh)
            stretched = auto_stretch(psf_data)

            channels.append(stretched)

        stacked = np.stack(channels, axis=0)
        return np.moveaxis(stacked, 0, -1)

    else:
        raise ValueError(f"Unknown stretch type: {stretch}")
