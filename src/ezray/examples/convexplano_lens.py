"""A EFL = +50.1 mm planoconvex lens: https://www.thorlabs.com/thorproduct.cfm?partnumber=LA1255

The object is at infinity; aperture stop is the first surface.

"""

from math import inf
from typing import Any

import numpy as np

from ezray import Axis, ParaxialModelID, OpticalSystem
from ezray.specs.aperture import EntrancePupil
from ezray.specs.fields import Angle
from ezray.specs.gaps import Gap
from ezray.specs.surfaces import Conic, Image, Object, SurfaceType


system: OpticalSystem = OpticalSystem(
    aperture=EntrancePupil(semi_diameter=12.5),
    fields=[Angle(angle=0, wavelength=0.5876), Angle(angle=5, wavelength=0.5876)],
    gaps=[
        Gap(thickness=inf),
        Gap(refractive_index=1.515, thickness=5.3),
        Gap(thickness=46.59874),
    ],
    surfaces=[
        Object(),
        Conic(
            semi_diameter=12.5,
            radius_of_curvature=25.8,
            surface_type=SurfaceType.REFRACTING,
        ),
        Conic(semi_diameter=12.5, surface_type=SurfaceType.REFRACTING),
        Image(),
    ],
)


_paraxial_properties = {
    "aperture_stop": 1,
    "back_focal_length": 46.59874,
    "back_principal_plane": 1.80174,
    "chief_ray": np.array(
        [
            [[0.0, 0.087489]],
            [[0.0, 0.0577482]],
            [[0.306067, 0.087489]],
            [[4.382944, 0.087489]],
        ]
    ),
    "effective_focal_length": 50.097,
    "entrance_pupil": {"location": 0.0, "semi_diameter": 12.5},
    "exit_pupil": {"location": 1.80165, "semi_diameter": 12.5},
    "front_focal_length": -50.097,
    "front_principal_plane": 0.0,
    "marginal_ray": np.array(
        [
            [[12.50000, 0]],
            [[12.50000, -0.16469]],
            [[11.62713, -0.24950]],
            [[0, -0.24950]],
        ]
    ),
    "paraxial_image_plane": {"location": 51.89874, "semi_diameter": 4.382944},
    "user_image_plane": {"location": 51.89874, "semi_diameter": 4.382944},
}

PARAXIAL_PROPERTIES: dict[ParaxialModelID, dict[str, Any]] = {
    (0.5876, Axis.X): _paraxial_properties,
    (0.5876, Axis.Y): _paraxial_properties,
}
