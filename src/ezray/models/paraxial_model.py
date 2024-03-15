"""Models for paraxial optical system design."""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, TypedDict

import numpy as np
from numpy.linalg import inv
import numpy.typing as npt

from ezray.core.general_ray_tracing import (
    Axis,
    Conic,
    Float,
    Image,
    Object,
    SequentialModel,
    Stop,
    Surface,
    SurfaceType,
    Toric,
)
from ezray.specs.fields import Angle, FieldSpec, ObjectHeight

"""A Ns x Nr x 2 array of ray trace results.

Ns is the number of surfaces, and Nr is the number of rays. The first column is the
height of the ray at the surface, and the second column is the angle of the ray at the
surface.

"""
type RayTraceResults = npt.NDArray[Float]


"""The thickness to assign a gap that is either None or infinite."""
DEFAULT_THICKNESS = 0.0


class RayFactory:
    """A factory for creating rays."""

    @staticmethod
    def ray(height: float = 0.0, angle: float = 0.0) -> npt.NDArray[Float]:
        """Return a ray with the given height and angle."""
        return np.array([height, angle])


def z_intercept(rays: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """Return the intercept of the rays with the z-axis.

    The intercept is the distance from the ray to the intercept, i.e. the origin is
    assumed to be at the point where the ray height is equal to the input.

    """
    rays = np.atleast_2d(rays)

    return -rays[:, 0] / rays[:, 1]


def propagate(rays: npt.NDArray[Float], distance: float) -> npt.NDArray[Float]:
    """Propagate rays a distance along the optical axis."""
    new_rays = np.atleast_2d(rays.copy())  # Copy to avoid modifying the input array.
    new_rays[:, 0] += distance * new_rays[:, 1]

    return new_rays


class ImagePlane(TypedDict):
    location: float
    semi_diameter: float


class Pupil(TypedDict):
    location: float
    semi_diameter: float


@dataclass(frozen=True)
class ParaxialModel:
    """A paraxial model of an optical system.

    Parameters
    ----------
    sequential_model : SequentialModel
        The sequential model of the optical system.
    fields : set[FieldSpec]
        The field specs of the optical system.
    object_space_telecentric : bool, optional
        If True, the system is telecentric in the object space. This forces the chief
        ray angles to be parallel to the axis in object space. Default is False.
    axis : Axis, optional
        The axis of the system to use for ray traces. Default is Axis.Y.

    """

    sequential_model: SequentialModel
    fields: set[FieldSpec]
    object_space_telecentric: bool = False
    axis: Axis = Axis.Y

    @cached_property
    def aperture_stop(self) -> int:
        """Returns the surface ID of the aperture stop.

        The aperture stop is the surface that has the smallest ratio of semi-diameter
        to ray height. If there are multiple surfaces with the same ratio, the first
        surface is returned.

        """
        results = self.pseudo_marginal_ray

        semi_diameters = np.array(
            [surface.semi_diameter for surface in self.sequential_model.surfaces]
        )
        ratios = semi_diameters / results[:, 0, 0].T

        # Do not include the object or image surfaces when finding the minimum.
        return np.argmin(ratios[1:-1]) + 1

    @cached_property
    def back_focal_length(self) -> float:
        """Returns the back focal length of the system."""
        results = self.parallel_ray

        last_real_surface_id = self.sequential_model.last_real_surface_id
        bfl = z_intercept(results[last_real_surface_id])[0]

        # I would expect an infinite BFL to be positive inf, but the intercept
        # calculation can be negative to cover all the non-infinite cases. Either
        # way, this is an edge case that is handled to make "intuitive sense."
        if np.isneginf(bfl):
            return np.inf

        return bfl

    @cached_property
    def back_principal_plane(self) -> float:
        """Returns the z-coordinate of the rear principal plane."""

        delta = self.back_focal_length - self.effective_focal_length

        # Principal planes make no sense for surfaces without power.
        if np.isinf(delta):
            return np.nan

        # Compute the z-position of the last real surface before the image plane.
        last_real_surface_id = self.sequential_model.last_real_surface_id
        z = self.z_coordinate(last_real_surface_id)

        return z + delta

    @cached_property
    def chief_ray(self) -> RayTraceResults:
        """Returns the chief ray through the system."""
        enp_loc: float = self.entrance_pupil["location"]
        obj_loc: float = 0 if self._is_obj_at_inf else self.z_coordinate(0)
        sep = enp_loc - obj_loc
        match max_field := max(abs(field) for field in self.fields):
            case Angle(angle=angle):
                paraxial_angle = np.tan(np.deg2rad(angle))
                height = -sep * paraxial_angle
            case ObjectHeight(height=height):
                # Object distance is finite with this field type by convention
                paraxial_angle = -height / sep
            case _:
                raise ValueError(f"Unknown field type: {max_field}")

        ray = RayFactory.ray(height=height, angle=paraxial_angle)

        return trace(ray, self.sequential_model, axis=self.axis)

    @cached_property
    def effective_focal_length(self) -> float:
        """Returns the effective focal length of the system."""
        results = self.parallel_ray

        y_1 = results[1, 0, 0]
        u_final = results[-2, 0, 1]
        efl = -y_1 / u_final

        # Handle edge case for infinite EFL
        if np.isneginf(efl):
            return np.inf

        return -y_1 / u_final

    @cached_property
    def entrance_pupil(self) -> Pupil:
        if self.object_space_telecentric:
            return {"location": np.inf, "semi_diameter": np.nan}

        # Aperture stop is first surface
        if self.aperture_stop == 1:
            return {
                "location": 0,
                "semi_diameter": self.sequential_model.surfaces[1].semi_diameter,
            }

        # Trace a ray from the aperture stop backwards through the system
        steps = self.sequential_model[: self.aperture_stop - 1]
        ray = RayFactory.ray(height=0.0, angle=-1.0)  # -1 to trace backwards

        results = trace(ray, steps, reverse=True, axis=self.axis)

        location = z_intercept(results[-1]).item()  # Relative to the first surface

        # Propagate marginal ray to the entrance pupil
        distance = (
            location
            if self._is_obj_at_inf
            else self.sequential_model.gaps[0].thickness + location
        )
        semi_diameter = propagate(self.marginal_ray[0, 0, :], distance)[0, 0]

        return {"location": location, "semi_diameter": semi_diameter}

    @cached_property
    def exit_pupil(self) -> Pupil:
        z_last_surface = self.z_coordinate(len(self.sequential_model.surfaces) - 2)

        # Aperture stop is last non-image plane surface.
        if self.aperture_stop == len(self.sequential_model.surfaces) - 2:
            return {
                "location": z_last_surface,
                "semi_diameter": self.sequential_model.surfaces[-2].semi_diameter,
            }

        # Trace a ray from the aperture stop forwards through the system
        steps = self.sequential_model[self.aperture_stop - 1 :]
        ray = RayFactory.ray(height=0.0, angle=1.0)

        results = trace(ray, steps, axis=self.axis)

        # Propagate marginal ray to the exit pupil
        distance = z_intercept(results[-2])  # Relative to the last surface
        semi_diameter = propagate(self.marginal_ray[-2, 0, :], distance)[0, 0]

        location = z_last_surface + distance

        return {"location": location, "semi_diameter": semi_diameter}

    @cached_property
    def front_focal_length(self) -> float:
        """Returns the front focal length of the system."""
        results = self.reversed_parallel_ray

        first_real_surface_id = self.sequential_model.first_real_surface_id
        results_id = self.sequential_model.reverse_id(first_real_surface_id)
        ffl = z_intercept(results[results_id])[0]

        if np.isneginf(ffl):
            return np.inf

        return ffl

    @cached_property
    def front_principal_plane(self) -> float:
        """Returns the z-coordinate of the front principal plane."""
        if np.isinf(self.front_focal_length):
            return np.nan

        return self.front_focal_length + self.effective_focal_length

    @cached_property
    def marginal_ray(self) -> RayTraceResults:
        """Returns the marginal ray through the system.

        By convention, the number of rays Nr is 1.

        """
        pmr = self.pseudo_marginal_ray

        semi_diameters = np.array(
            [surface.semi_diameter for surface in self.sequential_model.surfaces]
        )
        ratios = semi_diameters / pmr[:, 0, 0].T

        scale_factor = ratios[self.aperture_stop]

        return pmr * scale_factor

    @cached_property
    def parallel_ray(self) -> RayTraceResults:
        """A ray used to compute back focal lengths."""
        # Ray parallel to the optical axis at a height of 1.
        ray = RayFactory.ray(height=1.0, angle=0.0)

        return trace(ray, self.sequential_model, axis=self.axis)

    @cached_property
    def paraxial_image_plane(self) -> ImagePlane:
        """Returns the paraxial image plane.

        This is the theoretical image plane, not the user-defined one.

        See Also
        --------
        user_image_plane : The user-defined image plane.

        """
        dz = z_intercept(self.marginal_ray[-1])[0]
        location = self.z_coordinate(len(self.sequential_model.surfaces) - 1) + dz

        # Edge case for infinite image plane
        if np.isneginf(location):
            return {"location": np.inf, "semi_diameter": np.nan}

        # Take the chief ray and propagate it to the theoretical image plane.
        semi_diameter = abs(propagate(self.chief_ray[-1], dz)[0, 0])

        return {"location": location, "semi_diameter": semi_diameter}

    @cached_property
    def pseudo_marginal_ray(self) -> RayTraceResults:
        """Traces a pseudo-marginal ray through the system."""

        if self._is_obj_at_inf:
            # Ray parallel to the optical axis at a distance of 1.
            ray = RayFactory.ray(height=1.0, angle=0.0)
        else:
            # Ray originating at the optical axis at an angle of 1.
            ray = RayFactory.ray(height=0.0, angle=1.0)

        return trace(ray, self.sequential_model, axis=self.axis)

    @cached_property
    def reversed_parallel_ray(self) -> RayTraceResults:
        """A ray used to compute front focal lengths."""
        ray = RayFactory.ray(height=1.0, angle=0.0)

        return trace(ray, self.sequential_model, reverse=True, axis=self.axis)

    @cached_property
    def user_image_plane(self) -> ImagePlane:
        """Returns the location and semi-diameter of the user-defined image plane.

        This is a user-defined quantity because the image-space gap is provided by
        the user.

        See Also
        --------
        paraxial_image_plane : The theoretical image plane.

        """
        location = self.z_coordinate(len(self.sequential_model.surfaces) - 1)
        semi_diameter = abs(self.chief_ray[-1, 0, 0])

        return {"location": location, "semi_diameter": semi_diameter}

    def z_coordinate(self, surface_id: int) -> float:
        """Returns the z-coordinate of a surface.

        The origin is at the first surface.

        """
        if surface_id == 0 and np.isinf(self.sequential_model.gaps[0].thickness):
            return -np.inf
        if surface_id == 1:
            return 0.0

        return sum(gap.thickness for gap in self.sequential_model.gaps[1:surface_id])

    @cached_property
    def _is_obj_at_inf(self) -> bool:
        return np.isinf(self.sequential_model.gaps[0].thickness)


def surface_rtm_mapping(
    surface: Surface, reverse: bool = False
) -> Callable[[float, float, float, float], npt.NDArray[Float]]:
    """Return the ray transfer matrix for a surface."""
    if reverse:
        match surface:
            case Conic(surface_type=SurfaceType.REFRACTING):
                return lambda t, R, n0, n1: inv(
                    np.array([[1, 0], [(n1 - n0) / R / n0, n1 / n0]])
                ) @ np.array(
                    [[1, -t], [0, 1]]
                )  # n0 and n1 are swapped!
            case Conic(surface_type=SurfaceType.REFLECTING):
                return lambda t, R, *_: np.array([[1, 0], [2 / R, 1]]) @ np.array(
                    [[1, -t], [0, 1]]
                )
            case Image():
                return lambda *_: np.array([[1, 0], [0, 1]])
            case Stop():
                return lambda t, *_: np.array([[1, 0], [0, 1]]) @ np.array(
                    [[1, -t], [0, 1]]
                )
            case Object():
                return lambda t, *_: np.array([[1, 0], [0, 1]]) @ np.array(
                    [[1, -t], [0, 1]]
                )
            # Torics are treated the same as conics in the paraxial model.
            case Toric(surface_type=SurfaceType.REFRACTING):
                return lambda t, R, n0, n1: inv(
                    np.array([[1, 0], [(n0 - n1) / R / n0, n1 / n0]])
                ) @ np.array(
                    [[1, -t], [0, 1]]
                )  # n0 and n1 are swapped!
            case Toric(surface_type=SurfaceType.REFLECTING):
                return lambda t, R, *_: np.array([[1, 0], [2 / R, 1]]) @ np.array(
                    [[1, -t], [0, 1]]
                )
            case _:
                raise ValueError(f"Unknown surface type: {surface}")

    match surface:
        case Conic(surface_type=SurfaceType.REFRACTING):
            return lambda t, R, n0, n1: np.array(
                [[1, 0], [(n0 - n1) / R / n1, n0 / n1]]
            ) @ np.array([[1, t], [0, 1]])
        case Conic(surface_type=SurfaceType.REFLECTING):
            return lambda t, R, *_: np.array([[1, 0], [-2 / R, 1]]) @ np.array(
                [[1, t], [0, 1]]
            )
        case Image():
            return lambda t, *_: np.array([[1, 0], [0, 1]]) @ np.array([[1, t], [0, 1]])
        case Stop():
            return lambda t, *_: np.array([[1, 0], [0, 1]]) @ np.array([[1, t], [0, 1]])
        case Object():
            return lambda *_: np.array([[1, 0], [0, 1]])
        # Torics are treated the same as conics in the paraxial model.
        case Toric(surface_type=SurfaceType.REFRACTING):
            return lambda t, R, n0, n1: np.array(
                [[1, 0], [(n0 - n1) / R / n1, n0 / n1]]
            ) @ np.array([[1, t], [0, 1]])
        case Toric(surface_type=SurfaceType.REFLECTING):
            return lambda t, R, *_: np.array([[1, 0], [-2 / R, 1]]) @ np.array(
                [[1, t], [0, 1]]
            )
        case _:
            raise ValueError(f"Unknown surface type: {surface}")


def roc(surface: Surface, axis: Axis) -> float:
    """Return the radius of curvature of a surface along a given axis."""
    # By convention, the toric's radius of revolution is about the y-axis, so it is
    # the revolution curvature is xin the x-z plane.
    if isinstance(surface, Toric) and axis == Axis.X:
        return surface.radius_of_revolution

    return surface.radius_of_curvature


def rtms(
    steps: SequentialModel, reverse: bool = False, axis: Axis = Axis.Y
) -> list[npt.NDArray[Float]]:
    """Compute the ray transfer matrices for each tracing step.

    In the case that the object space is of infinite extent, the first gap thickness is
    set to a default, finite value.

    Parameters
    ----------
    steps : Iterable[TracingStep]
        An iterable of ray tracing steps.
    reverse : bool, optional
        If True, the ray transfer matrices are computed for a ray trace in the reverse
        direction. This is useful, for example, for computing the entrance pupil
        location.

    """
    if reverse:
        steps = [
            (gap_1, surface, gap_0) for gap_0, surface, gap_1 in reversed(tuple(steps))
        ]

    txs = []
    for gap_0, surface, gap_1 in steps:
        t = (
            DEFAULT_THICKNESS
            if gap_0 is None or np.isinf(gap_0.thickness)
            else gap_0.thickness
        )
        R = roc(surface, axis)
        n0 = gap_1.refractive_index if gap_0 is None else gap_0.refractive_index
        n1 = gap_0.refractive_index if gap_1 is None else gap_1.refractive_index

        txs.append(surface_rtm_mapping(surface, reverse=reverse)(t, R, n0, n1))

    return txs


def trace(
    rays: npt.NDArray[Float],
    steps=SequentialModel,
    reverse=False,
    axis=Axis.Y,
) -> npt.NDArray[Float]:
    """Trace rays through a paraxial system.

    Parameters
    ----------
    rays : npt.NDArray[Float]
        Array of rays to trace through the system. Each row is a ray, and the
        columns are the ray height and angle.
    reverse : bool, optional
        If True, the rays are traced in the reverse direction. This is useful, for
        example, for computing the entrance pupil location.

    """
    # Ensure that the rays are a 2D array.
    rays = np.atleast_2d(rays)

    # Compute the ray transfer matrices for each step.
    txs = rtms(steps, reverse=reverse, axis=axis)

    # Pre-allocate the results. Shape is Ns X Nr X 2, where Ns is the number of
    # surfaces, Nr is the number of rays, and 2 is the ray height and angle.
    results = np.empty((len(txs) + 1, rays.shape[0], 2))
    results[0] = rays

    # Trace the rays through the system.
    for i, tx in enumerate(txs):
        rays = (tx @ rays.T).T
        results[i + 1] = rays

    return results
