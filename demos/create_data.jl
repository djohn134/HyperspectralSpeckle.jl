include("../src/HyperspectralSpeckle.jl")
using Main.HyperspectralSpeckle
using Statistics

############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "test"
id = ""
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64

nsubaps_side = 6
nλ = 1
λ_ref = 500.0e-9  # m
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9  # m
λ = collect(range(λmin, stop=λmax, length=nλ))  # m
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)  # m
resolution = mean(λ) / Δλ
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
D_inner_frac = 0.0
aperture_area = pi * (D / 2)^2 * (1 - D_inner_frac^2)
fov = 10.0  # arcsec
pixscale_full = fov / image_dim  # arcsec/pix
pixscale_wfs = pixscale_full .* nsubaps_side  # arcsec/pix
DTYPE = UInt16
qefile = "../data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)  # e⁻ / ph
rn = 2.0  # e⁻
saturation = 30000.0  # e⁻
gain = saturation / (typemax(DTYPE))  # e⁻ / ADU
DTYPE = FTYPE
exptime = 5e-3  # sec
nepochs = 11
times = collect(0:nepochs-1) .* exptime
noise = true
ζ = 0.0  # deg
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    gain=gain,
    saturation=saturation,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE,
    DTYPE=DTYPE,
    label="Full Aperture"
)
# diversity = Diversity(
#     4,
#     1.0,
#     2*(maximum(times)-minimum(times)),
#     minimum(times),
#     verb=verb,
#     FTYPE=FTYPE
# )

### Create Full-Ap Observations object ####
# ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
# ϕ_static_full = repeat(masks_full.masks[:, :, 1, 1] .* create_zernike_screen(image_dim, image_dim÷4, 4, 4.0, FTYPE=FTYPE), 1, 1, nλ)
# writefits(ϕ_static_full, "$(folder)/defocus4.fits")
observations_full = Observations(
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    D_inner_frac=D_inner_frac,
    times=times,
    nsubexp=1,
    nsubaps_side=1,
    dim=image_dim,
    # diversity=diversity,
    verb=verb,
    FTYPE=FTYPE,
    label="Full Aperture"
)
# observations = [observations_full]
optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    gain=gain,
    saturation=saturation,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE,
    DTYPE=DTYPE,
    label="Wavefront Sensor"
)
observations_wfs = Observations(
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    area=aperture_area,
    times=times,
    nsubexp=1,
    nsubaps_side=nsubaps_side,
    dim=wfs_dim,
    # ϕ_static=ϕ_static_wfs,
    build_dim=observations_full.dim,
    verb=verb,
    FTYPE=FTYPE,
    label="Wavefront Sensor"
)
nsubaps = observations_wfs.masks.nsubaps
observations_wfs.masks.scale_psfs = observations_full.masks.scale_psfs
observations = [observations_full, observations_wfs]
header = create_header(λ, units="unitless")
[writefits(observations[dd].masks.masks, "$(folder)/masks_ISH$(observations[dd].masks.nsubaps_side)x$(observations[dd].masks.nsubaps_side)$(id).fits", header=header) for dd=1:length(observations)]
###########################################

############ Object Parameters ############
objectfile = "../data/hubble_truth2.fits"
# object_arr, ~ = template2object(objectfile, image_dim, λ, FTYPE=FTYPE)
object_arr = repeat(block_reduce(readfits(objectfile, FTYPE=FTYPE), image_dim), 1, 1, nλ)
~, spectrum = solar_spectrum(λ=λ)
mag = 4.0
background_mag = 20.0  # mag / arcsec^2
irradiance = mag2flux(mag, filter, ζ=ζ)  # ph / s / m^2
background = mag2flux(background_mag, filter, ζ=ζ)  # ph / s / m^2 / arcsec^2
background *= fov^2  # ph / s / m^2
object_range = 500.0e3  # m
############## Create object ##############
object = Object(
    object_arr,
    irradiance=irradiance,
    background=background,
    λ=λ,
    fov=fov,
    object_range=object_range, 
    dim=image_dim,
    spectrum=spectrum,
    verb=verb,
    FTYPE=FTYPE
)
header = create_header(λ, units="ph/s/m^2/m")
writefits(object.object, "$(folder)/object_truth_spectral$(id).fits", header=header)
header = create_header(λ, units="ph/s/m^2")
writefits(dropdims(sum(object.object, dims=3), dims=3)*Δλ, "$(folder)/object_truth$(id).fits", header=header)
###########################################

########## Anisopatch Parameters ##########
isoplanatic = false
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
###########################################

########## Atmosphere Parameters ##########
Dr0_ref_vertical = 20.0
heights = [0.0, 7000.0, 12500.0]  # m
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0, 125.0, 135.0]  # deg
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
seeds = [713, 1212, 525118]
Dmeta = D .+ (fov/206265) .* heights
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("/home/dan/Desktop/HyperspectralSpeckle.jl/data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    object, 
    patches,
    Dr0=Dr0_ref_vertical,
    wind=wind, 
    heights=heights, 
    transmission=transmission,
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    propagate=propagate,
    seeds=seeds, 
    FTYPE=FTYPE,
    verb=verb
)
########## Create phase screens ###########
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
header = create_header(λ, units="rad")
writefits(atmosphere.phase, "$(folder)/Dr0_$(round(Int64, Dr0_ref_vertical))_phase_full$(id).fits", header=header)
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
###########################################

########## Create Full-Ap images ##########
# using BenchmarkTools
# @btime [create_detector_images($patches, $observations[dd], $atmosphere, $object, build_dim=$image_dim, noise=$noise, verb=$verb) for dd=1:length(observations)]
[create_detector_images(patches, observations[dd], atmosphere, object, build_dim=image_dim, noise=noise, verb=verb) for dd=1:length(observations)]
[writefits(observations[dd].images, "$(folder)/Dr0_$(round(Int64, Dr0_ref_vertical))_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_images$(id).fits", header=create_header(observations[dd]), times=observations[dd].times) for dd=1:length(observations)]
###########################################

## Isoplanatic
# Naive threadix: 1.278 s (546486 allocations: 137.16 MiB)
# OhMyThreads.Channel: 1.264 s (533701 allocations: 144.43 MiB)
#                      1.240 s (536749 allocations: 144.49 MiB)
## Anisoplanatic
# Naive threadix: 197.684 s (751385 allocations: 144.23 MiB)
# OhMyThreads.Channel: 205.700 s (748062 allocations: 151.27 MiB)
#                      102.606 s (743478 allocations: 151.16 MiB)   