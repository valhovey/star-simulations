import numpy as np
import matplotlib.pyplot as plt
import poppy
from poppy import ArrayOpticalElement, OpticalSystem
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap

print("POPPY version:", poppy.__version__)

npix = 1024
aperture_radius = 0.1 * u.m
secondary_radius = 0.03 * u.m
support_width = 0.0015 * u.m
support_offset_max = 0.02 * u.m
frames = 200

d_offset = support_offset_max / frames

for i, support_offset in enumerate([0 + n * d_offset for n in range(frames)]):
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

    pixelscale = 2*aperture_radius/(npix * u.pixel)
    array_pupil = ArrayOpticalElement(
            transmission=pupil,
            pixelscale=pixelscale,
    )

    osys = OpticalSystem(pupil_diameter=2*aperture_radius)
    osys.add_pupil(array_pupil)

    detector_pixelscale = 0.1 * u.arcsec / u.pixel  # arcsec per pixel
    osys.add_detector(pixelscale=detector_pixelscale, fov_pixels=512)

    colors = [
        np.arange(620e-9, 700e-9, 10e-9),
        np.arange(480*1e-9, 570*1e-9, 10*1e-9),
        np.arange(420*1e-9, 510*1e-9, 10*1e-9),
    ]

    eps = 1e-12

    # Store log-scaled PSFs here
    log_psf_channels = []

    for wavelengths in colors:
        weights = np.ones_like(wavelengths)
        source = {"wavelengths": wavelengths, "weights": weights}

        psf = osys.calc_psf(source=source)
        psf_data = psf[0].data

        # Safe log scale
        log_psf = np.log10(psf_data + eps)
        log_psf_channels.append(log_psf)

    # Convert list → stacked array, shape: (3, H, W)
    log_psf_channels = np.array(log_psf_channels)

    # Normalize each channel 0–1 for display
    # (Use same global min/max so colors are balanced)
    min_val = log_psf_channels.min()
    max_val = log_psf_channels.max()
    norm = (log_psf_channels - min_val) / (max_val - min_val)

    # Rearrange to (H, W, 3) for RGB
    rgb_image = np.moveaxis(norm, 0, -1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ------------------------
    # Left pane: Aperture
    # ------------------------
    ax = axes[0]
    ax.set_xticks([])
    ax.set_yticks([])
    im0 = ax.imshow(
        pupil,
        origin='lower',
        cmap='gray',
        extent=[-aperture_radius.value, aperture_radius.value,
                -aperture_radius.value, aperture_radius.value]
    )
    ax.set_title("Spider Shape")

    # ------------------------
    # Right pane: PSF (RGB)
    # ------------------------
    ax = axes[1]
    ax.imshow(rgb_image, origin='lower')
    ax.set_title("RGB PSF")
    ax.axis('off')

    plt.tight_layout()

    # Save instead of show
    fig.savefig(f"frame_{i}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)
