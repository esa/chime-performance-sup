"""Atmospheric quantities and the Lambertian surface-atmosphere equations."""

import json
from importlib.metadata import version

import numpy as np
import xarray as xr
from eopl import image_t

from chime_perf.algorithms.algorithms import require_units


def prepare_atmosphere(
    wavelength_nm,
    sixs_path=None,
    solar_zenith_deg=30.0,
    view_zenith_deg=0.0,
    solar_azimuth_deg=0.0,
    view_azimuth_deg=0.0,
    month=6,
    day=21,
    aot550=0.1,
    target_altitude_km=0.0,
    profile="USStandard1962",
    aerosol="Continental",
):
    """Obtain gas/scattering transmission, path radiance, albedo and solar irradiance."""
    from Py6S import AeroProfile, AtmosProfile, Geometry, GroundReflectance, SixS, Wavelength

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)

    if wavelength_nm.ndim != 1 or not np.isfinite(wavelength_nm).all():
        raise ValueError("Wavelengths must be a finite 1-D array.")

    if len(wavelength_nm) == 0 or np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("Wavelengths must be nonempty and strictly increasing.")

    if wavelength_nm[0] < 250 or wavelength_nm[-1] > 4000:
        raise ValueError("6S wavelengths must lie between 250 and 4000 nm.")

    if not 0 <= solar_zenith_deg < 90 or not 0 <= view_zenith_deg < 90 or aot550 < 0:
        raise ValueError(
            "Zenith angles must be in [0, 90); aerosol thickness must be nonnegative."
        )

    model = SixS(sixs_path)

    if model.sixs_path is None:
        raise RuntimeError("6S executable not found; install 6S or supply --sixs-path.")

    model.atmos_profile = AtmosProfile.PredefinedType(getattr(AtmosProfile, profile))
    model.aero_profile = AeroProfile.PredefinedType(getattr(AeroProfile, aerosol))
    model.aot550 = aot550

    model.geometry = Geometry.User()
    model.geometry.solar_z = solar_zenith_deg
    model.geometry.solar_a = solar_azimuth_deg
    model.geometry.view_z = view_zenith_deg
    model.geometry.view_a = view_azimuth_deg
    model.geometry.month = month
    model.geometry.day = day

    model.altitudes.set_target_custom_altitude(target_altitude_km)
    model.altitudes.set_sensor_satellite_level()

    # A black surface isolates the atmospheric contribution to the sensor radiance.
    model.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    # 6SV1.1 uses a 2.5 nm grid. Include neighbours on both sides of the scene range.
    first = int(np.floor(wavelength_nm[0] / 2.5))
    last = int(np.ceil(wavelength_nm[-1] / 2.5))
    solver_nm = np.arange(first, last + 1) * 2.5

    gas = np.empty(len(solver_nm))
    downward = np.empty(len(solver_nm))
    upward = np.empty(len(solver_nm))
    path = np.empty(len(solver_nm))
    albedo = np.empty(len(solver_nm))
    irradiance = np.empty(len(solver_nm))

    for channel, wavelength in enumerate(solver_nm):
        model.wavelength = Wavelength(wavelength / 1000)  # 6S takes micrometres.
        model.run()
        output = model.outputs

        gas[channel] = output.total_gaseous_transmittance
        downward[channel] = output.transmittance_total_scattering.downward
        upward[channel] = output.transmittance_total_scattering.upward
        path[channel] = output.atmospheric_intrinsic_radiance
        albedo[channel] = output.spherical_albedo.total
        irradiance[channel] = output.solar_spectrum

    raw = {
        "gas_transmission": gas,
        "downward_transmission": downward,
        "upward_transmission": upward,
        "path_radiance": path,
        "spherical_albedo": albedo,
        "solar_irradiance": irradiance,
    }
    quantities = xr.Dataset(coords={"SPE": wavelength_nm})

    for name, values in raw.items():
        if not np.isfinite(values).all():
            raise ValueError(f"6S returned non-finite {name}; inspect the solver setup.")

        # Interpolating these quantities does not add additional atmospheric information.
        quantities[name] = ("SPE", np.interp(wavelength_nm, solver_nm, values))
        quantities[name].attrs["units"] = "1"

    quantities.SPE.attrs["units"] = "nm"
    quantities.path_radiance.attrs["units"] = "W m-2 sr-1 um-1"
    quantities.solar_irradiance.attrs["units"] = "W m-2 um-1"

    settings = {
        "solar_zenith_deg": solar_zenith_deg,
        "view_zenith_deg": view_zenith_deg,
        "solar_azimuth_deg": solar_azimuth_deg,
        "view_azimuth_deg": view_azimuth_deg,
        "month": month,
        "day": day,
        "aot550": aot550,
        "target_altitude_km": target_altitude_km,
        "profile": profile,
        "aerosol": aerosol,
    }
    quantities.attrs = {
        "solver": "6S",
        "py6s_version": version("Py6S"),
        "sixs_version": str(model.outputs.version),
        "sixs_executable": str(model.sixs_path),
        "solver_step_nm": 2.5,
        "solar_zenith_deg": solar_zenith_deg,
        "settings": json.dumps(settings),
    }

    return quantities


def combine_transmission(gas, downward, upward):
    """T = t_gas * t_down * t_up, including absorption and both scattering paths."""
    for quantity in (gas, downward, upward):
        require_units(quantity.image(), "1")

    gas_values, down_values, up_values = xr.align(
        gas.image(), downward.image(), upward.image(), join="exact"
    )

    transmission = gas_values * down_values * up_values
    transmission = gas.image().copy(data=transmission.data)
    transmission.attrs["units"] = "1"

    return image_t(transmission)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
combine_transmission.cuda_support = False
combine_transmission.dask_support = False
combine_transmission.input_access = ("read", "read", "read")


def calculate_solar_scale(irradiance, solar_zenith_deg):
    """C = E_0 * cos(solar zenith) / pi, converting irradiance to a radiance scale."""
    require_units(irradiance.image(), "W m-2 um-1")
    zenith_rad = np.deg2rad(solar_zenith_deg)

    scale = irradiance.image() * np.cos(zenith_rad) / np.pi
    scale = irradiance.image().copy(data=scale.data)
    scale.attrs["units"] = "W m-2 sr-1 um-1"

    return image_t(scale)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
calculate_solar_scale.cuda_support = False
calculate_solar_scale.dask_support = False
calculate_solar_scale.input_access = ("read",)


def propagate(data, atmosphere):
    """L = P + C T rho / (1 - S rho), on the reflectance cube's coordinates."""
    reflectance = data.image()

    if reflectance.attrs.get("units") != "1":
        raise ValueError("Input must be BOA reflectance with units='1'.")

    if atmosphere.SPE.attrs.get("units") != "nm":
        raise ValueError("Atmospheric SPE coordinates must use nm.")

    reflectance, atmosphere = xr.align(reflectance, atmosphere, join="exact")

    units = {
        "transmission": "1",
        "spherical_albedo": "1",
        "path_radiance": "W m-2 sr-1 um-1",
        "solar_scale": "W m-2 sr-1 um-1",
    }
    for name, unit in units.items():
        if atmosphere[name].dims != ("SPE",) or atmosphere[name].attrs.get("units") != unit:
            raise ValueError(f"Atmospheric {name} must be a SPE array with units={unit!r}.")

        if not np.isfinite(atmosphere[name]).all():
            raise ValueError(f"Atmospheric {name} must be finite.")

    transmission = atmosphere["transmission"]
    path = atmosphere["path_radiance"]
    albedo = atmosphere["spherical_albedo"]
    scale = atmosphere["solar_scale"]

    if bool(((transmission < 0) | (albedo < 0) | (albedo >= 1) | (scale <= 0)).any()):
        raise ValueError("Need transmission >= 0, 0 <= albedo < 1 and solar scale > 0.")

    # Missing scene values and nonphysical denominators stay masked as NaN.
    denominator = 1 - albedo * reflectance
    valid = np.isfinite(reflectance) & (reflectance >= 0) & (denominator > 0)

    radiance = path + scale * transmission * reflectance / denominator.where(valid)
    radiance = reflectance.copy(data=radiance.transpose(*reflectance.dims).data)
    radiance.attrs = {
        **reflectance.attrs,
        "units": "W m-2 sr-1 um-1",
        "product": "TOA radiance",
    }

    return image_t(radiance)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
propagate.cuda_support = False
propagate.dask_support = False
propagate.input_access = ("read",)
