import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import simulation as sim
from typing import cast

npix = 1024
aperture_radius = cast(u.Quantity, 0.01 * u.m)
secondary_radius = cast(u.Quantity, 0.003 * u.m)
support_width = cast(u.Quantity, 0.0005 * u.m)
support_offset_max = cast(u.Quantity, 0.0015 * u.m)
frames = 200

d_offset = support_offset_max / frames

def gen_frame(i, support_offset):
    pupil = sim.Pupil(aperture_radius, npix)
    pupil.add_secondary(secondary_radius)
    pupil.add_offset_supports(support_width, support_offset)

    osys = pupil.to_optical_system(0.7)

    rgb_image = sim.simulate_psf(sim.rgb, osys)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        pupil.pupil,
        origin='lower',
        cmap='gray',
        extent=[-aperture_radius.value, aperture_radius.value,
                -aperture_radius.value, aperture_radius.value]
    )
    ax.set_title("Spider Shape")

    ax = axes[1]
    ax.imshow(rgb_image, origin='lower')
    ax.set_title("RGB PSF")
    ax.axis('off')

    plt.tight_layout()

    fig.savefig(f"frame_{i}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)

if __name__ == "__main__":
    # Single shot
    gen_frame(0, 0)

    # Animation
    # for i, support_offset in enumerate([0 + n * d_offset for n in range(frames)]):
    #     gen_frame(i, support_offset)
