import numpy as np
import matplotlib.pyplot as plt
import poppy
from poppy import ArrayOpticalElement, OpticalSystem
import astropy.units as u

print("POPPY version:", poppy.__version__)

npix = 512
aperture_radius = 0.05 * u.m
secondary_radius = 0.01 * u.m
support_width = 0.002 * u.m
support_offset = 0.000 * u.m

x = np.linspace(-aperture_radius.value, aperture_radius.value, npix) * u.m
y = np.linspace(-aperture_radius.value, aperture_radius.value, npix) * u.m
xx, yy = np.meshgrid(x, y)

pupil = ((xx**2 + yy**2) <= aperture_radius**2).astype(float)
pupil *= ((xx**2 + yy**2) >= secondary_radius**2).astype(float)

directions = [(0, xx, yy), (np.pi/2, yy, xx)]
for (angle, left, up) in directions:
    x_rot = xx * np.cos(angle) + yy * np.sin(angle)
    centered = ((xx**2 + yy**2) <= aperture_radius**2)
    l_mask = (np.abs(x_rot - support_offset) < support_width/2) & centered & (left > -support_width/2) & (up > 0)
    r_mask = (np.abs(x_rot + support_offset) < support_width/2) & centered & (left < support_width/2) & (up < 0)
    pupil[l_mask | r_mask] = 0.0

plt.figure(figsize=(6,6))
plt.imshow(pupil, origin='lower', cmap='gray',
           extent=[-aperture_radius.value, aperture_radius.value,
                   -aperture_radius.value, aperture_radius.value])
plt.colorbar(label='Amplitude')
plt.title("Danostar Spider")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()

pixelscale = 2*aperture_radius/(npix * u.pixel)
array_pupil = ArrayOpticalElement(
        transmission=pupil,
        pixelscale=pixelscale,
)

osys = OpticalSystem(pupil_diameter=2*aperture_radius)
osys.add_pupil(array_pupil)

detector_pixelscale = (1.41 / 3) * u.arcsec / u.pixel  # arcsec per pixel
osys.add_detector(pixelscale=detector_pixelscale, fov_pixels=500)

colors = [
    np.arange(620e-9, 700e-9, 10e-9),
    np.arange(480*1e-9, 570*1e-9, 10*1e-9),
    np.arange(420*1e-9, 510*1e-9, 10*1e-9),
]

for wavelengths in colors:
    weights = np.ones_like(wavelengths)
    source = {"wavelengths": wavelengths, "weights": weights}

    psf = osys.calc_psf(source=source)

    plt.figure(figsize=(6,6))
    poppy.display_psf(psf, title="PSF with 4 Pairs of Parallel Supports")
    plt.show()
