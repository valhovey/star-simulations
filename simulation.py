from typing import Any
import numpy as np
import astropy.units as u
from dataclasses import dataclass
from numpy.typing import NDArray
from poppy import ArrayOpticalElement, OpticalSystem

@dataclass
class Pupil:
    radius: u.Quantity
    pixels: int
    xx: NDArray[Any]
    yy: NDArray[Any]
    pupil: NDArray[Any]

    def __init__(self, radius: u.Quantity, pixels: int):
        x = np.linspace(-radius.value, radius.value, pixels) * u.m
        self.radius = radius
        self.pixels = pixels
        self.xx, self.yy = np.meshgrid(x, x)
        self.pupil = ((self.xx**2 + self.yy**2) <= radius**2).astype(float)

    def add_secondary(self, radius):
        self.pupil *= ((self.xx**2 + self.yy**2) >= radius**2).astype(float)

    def add_offset_supports(self, width: u.Quantity, offset: u.Quantity):
        directions = [(0, self.xx, self.yy), (np.pi/2, self.yy, self.xx)]
        for (angle, left, up) in directions:
            x_rot = self.xx * np.cos(angle) + self.yy * np.sin(angle)
            centered = ((self.xx**2 + self.yy**2) <= self.radius**2)
            l_mask = (np.abs(x_rot - offset) < width/2) & centered & (left > -width/2) & (up > 0)
            r_mask = (np.abs(x_rot + offset) < width/2) & centered & (left < width/2) & (up < 0)
            self.pupil[l_mask | r_mask] = 0.0

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

red_high = 620e-9
red_low = 700e-9
green_high = 480e-9
green_low = 570e-9
blue_high=420e-9
blue_low=510e-9

def spectrum(low, high, dwave=10e-9):
    return np.arange(low, high, dwave)

rgb = [
    spectrum(red_high, red_low),
    spectrum(green_high, blue_low, 10*1e-9),
    spectrum(blue_high, blue_low, 10*1e-9),
]

def simulate_psf(colors, osys):
    eps = 1e-12

    log_psf_channels = []

    for wavelengths in colors:
        source = {
                "wavelengths": wavelengths,
                "weights": np.ones_like(wavelengths)
        }

        psf = osys.calc_psf(source=source)
        psf_data = psf[0].data

        log_psf = np.log10(psf_data + eps)
        log_psf_channels.append(log_psf)

    log_psf_channels = np.array(log_psf_channels)

    min_val = log_psf_channels.min()
    max_val = log_psf_channels.max()
    norm = (log_psf_channels - min_val) / (max_val - min_val)

    return np.moveaxis(norm, 0, -1)
