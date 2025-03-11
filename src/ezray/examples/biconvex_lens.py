"""A EFL = +100 mm biconvex lens: https://www.thorlabs.com/thorproduct.cfm?partnumber=LB1676-A

The object is at 200 mm from the front surface; aperture stop is the first surface.

"""

from typing import Any

import numpy as np

from ezray import Axis, ParaxialModelID, OpticalSystem
from ezray.specs.aperture import EntrancePupil
from ezray.specs.fields import ObjectHeight
from ezray.specs.gaps import Gap
from ezray.specs.surfaces import Conic, Image, Object, SurfaceType


system: OpticalSystem = OpticalSystem(
    aperture=EntrancePupil(semi_diameter=12.7),
    fields=[
        ObjectHeight(height=0, wavelength=0.5876),
        ObjectHeight(height=5, wavelength=0.5876),
    ],
    gaps=[
        Gap(thickness=200),
        Gap(refractive_index=1.517, thickness=3.6),
        Gap(thickness=196.1684),
    ],
    surfaces=[
        Object(),
        Conic(
            semi_diameter=12.7,
            radius_of_curvature=102.4,
            surface_type=SurfaceType.REFRACTING,
        ),
        Conic(
            semi_diameter=12.7,
            radius_of_curvature=-102.4,
            surface_type=SurfaceType.REFRACTING,
        ),
        Image(),
    ],
)


_paraxial_properties = {
    "aperture_stop": 1,
    "back_focal_length": 98.4360,
    "back_principal_plane": 2.4063,
    "chief_ray": np.array(
        [
            [[5.0, -0.025]],
            [[0.0, -0.01648]],
            [[-0.0593, -0.02470]],
            [[-4.9043, -0.02470]],
        ]
    ),
    "effective_focal_length": 99.6297,
    "entrance_pupil": {"location": 0.0, "semi_diameter": 12.7},
    "exit_pupil": {"location": 1.1981, "semi_diameter": 12.8540},
    "front_focal_length": -98.4360,
    "front_principal_plane": 1.1937,
    "marginal_ray": np.array(
        [
            [[0, 0.0635]],
            [[12.70000, -0.0004088]],
            [[12.6985, -0.06473]],
            [[0.0, -0.06473]],
        ]
    ),
    "paraxial_image_plane": {"location": 199.7684, "semi_diameter": 4.9048},
    "user_image_plane": {"location": 199.7684, "semi_diameter": 4.9048},
}

PARAXIAL_PROPERTIES: dict[ParaxialModelID, dict[str, Any]] = {
    (0.5876, Axis.X): _paraxial_properties,
    (0.5876, Axis.Y): _paraxial_properties,
}
