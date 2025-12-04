import matplotlib.pyplot as plt
import astropy.units as u
import simulation as sim
import numpy as np
from typing import cast

npix = 1024
aperture_radius = cast(u.Quantity, 0.1 * u.m)
secondary_radius = cast(u.Quantity, 0.03 * u.m)
support_width = cast(u.Quantity, 0.0015 * u.m)
support_offset_max = cast(u.Quantity, 0.00075 * u.m)
misalignment_max = 0.002
frames = 200

d_offset = support_offset_max / frames
d_misalignment = misalignment_max / frames

def gen_frame(i, support_offset, misalignment):
    pupil = sim.Pupil(aperture_radius, npix)
    pupil.add_secondary(secondary_radius)
    pupil.add_offset_supports(support_width, support_offset, misalignment)

    # monochrome
    # osys = pupil.to_optical_system(arcsec_per_pixel=0.3, fov_pixels=400)
    # rgb_image = sim.simulate_psf([[sim.red_low], [sim.red_low], [sim.red_low]], osys)

    # rgb (oversampled)
    osys = pupil.to_optical_system(arcsec_per_pixel=0.7, fov_pixels=600)
    rgb_image = sim.simulate_psf(sim.rgb, osys, sim.StretchType.LOG)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    pupil.display_on(axes[0])

    ax = axes[1]
    ax.imshow(rgb_image, origin='lower')
    ax.set_title("RGB PSF")
    ax.axis('off')

    plt.tight_layout()

    fig.savefig(f"frame_{i}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)

if __name__ == "__main__":
    # Single shot
    gen_frame(
            0,
            support_offset=0.000 * u.m,
            misalignment=0
    )

    # Animation
    # for i, misalignment in enumerate([0 + n * d_misalignment for n in range(frames)]):
    #     gen_frame(i, support_offset=0.010 * u.m, misalignment=misalignment)
