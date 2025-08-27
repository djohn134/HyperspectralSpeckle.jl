<<<<<<< HEAD
include("../src/HyperspectralSpeckle.jl")
using Main.HyperspectralSpeckle
using Statistics
=======
include("../src/mfbd.jl")
using Main.MFBD
>>>>>>> main

using Statistics
############# Data Parameters #############
FTYPE = Float32;
<<<<<<< HEAD
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "test"
=======
folder = "data/test"
>>>>>>> main
id = ""
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 32

<<<<<<< HEAD
nsubaps_side = 6
nλ = 1
λ_ref = 500.0e-9  # m
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9  # m
λ = collect(range(λmin, stop=λmax, length=nλ))  # m
# λ_nyquist = minimum(λ)  # m
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
noise = false
ζ = 0.0  # deg
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
=======
nsubaps_side = 12
nepochs = 100
nλ = 1
λ_nyquist = 500.0
λ_ref = 500.0
λmin = 500.0
λmax = 500.0
λ = collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)

header = (
    ["WAVELENGTH_START", "WAVELENGTH_END", "WAVELENGTH_STEPS"], 
    [λmin, λmax, nλ], 
    ["Shortest wavelength of mask [nm]", "Largest wavelength of mask [nm]", "Number of wavelength steps"]
)
########## Create Full-Ap Masks ###########
masks_full = Masks(
    dim=image_dim,
    nsubaps_side=1, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
maskfile = "$(folder)/ish_subaps_1x1$(id).fits"
writefits(masks_full.masks, maskfile, header=header)
masks_wfs = Masks(
    dim=image_dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
maskfile = "$(folder)/ish_subaps_$(nsubaps_side)x$(nsubaps_side)$(id).fits"
writefits(masks_wfs.masks, maskfile, header=header)
masks = [masks_full, masks_wfs]
# masks = [masks_full]
#########################################

### Detector & Observations Parameters ####
D = 3.6  # m
# fov = 20 * 256 / (132 * (256 / 512))
fov = 20.0
# fov = 100.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full * nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 2.0
exptime = 5e-3
noise = false
ζ = 0.0
>>>>>>> main
######### Create Full-Ap Detector #########
filter = Filter(filtername="Bessell:V", FTYPE=FTYPE)
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    gain=gain,
    saturation=saturation,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
<<<<<<< HEAD
    verb=verb,
    FTYPE=FTYPE,
    DTYPE=DTYPE,
    label="Full Aperture"
=======
    filter=filter,
    FTYPE=FTYPE
>>>>>>> main
)
### Create Full-Ap Observations object ####
observations_full = Observations(
    detector_full,
    ζ=ζ,
    D=D,
    D_inner_frac=D_inner_frac,
    times=times,
    nsubexp=1,
    nsubaps_side=1,
    dim=image_dim,
<<<<<<< HEAD
    ϕ_static=ϕ_static_full,
    verb=verb,
    FTYPE=FTYPE,
    label="Full Aperture"
)
# observations = [observations_full]
optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
=======
    α=0.5,
    FTYPE=FTYPE
)
>>>>>>> main
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    gain=gain,
    saturation=saturation,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
<<<<<<< HEAD
    verb=verb,
    FTYPE=FTYPE,
    DTYPE=DTYPE,
    label="Wavefront Sensor"
=======
    filter=filter,
    FTYPE=FTYPE
>>>>>>> main
)
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
<<<<<<< HEAD
    area=aperture_area,
    times=times,
    nsubexp=1,
    nsubaps_side=nsubaps_side,
    dim=wfs_dim,
    ϕ_static=ϕ_static_wfs,
    build_dim=observations_full.dim,
    verb=verb,
    FTYPE=FTYPE,
    label="Wavefront Sensor"
=======
    nepochs=nepochs,
    nsubaps=masks_wfs.nsubaps,
    dim=wfs_dim,
    α=0.5,
    FTYPE=FTYPE
>>>>>>> main
)
nsubaps = observations_wfs.masks.nsubaps
observations_wfs.masks.scale_psfs = observations_full.masks.scale_psfs
observations = [observations_full, observations_wfs]
<<<<<<< HEAD
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
=======
# observations = [observations_full]
###########################################

############ Object Parameters ############
objectfile = "data/star.fits"
~, spectrum = solar_spectrum(λ=λ)
# spectrum = ones(FTYPE, nλ)
template = false
mag = 4
background_mag = Inf ## mag / arcsec^2
flux = mag2flux(λ, spectrum, mag, observations_full.detector, D=D, ζ=ζ, exptime=exptime)
background_flux = mag2flux(λ, ones(nλ), background_mag, observations_full.detector, D=D, ζ=ζ, exptime=exptime)
background_flux *= fov^2
object_height = Inf# 1.434e6  # km
>>>>>>> main
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
<<<<<<< HEAD
    verb=verb,
=======
    qe=qe,
    objectfile=objectfile, 
    template=template,
>>>>>>> main
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
<<<<<<< HEAD
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
~, transmission = readtransmission("../data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    object, 
    patches,
    Dr0=Dr0_ref_vertical,
=======
l0 = 0.01  # m
L0 = 100.0  # m
Dr0_vertical = 20.0
Dr0_composite = Dr0_vertical * sec(ζ*pi/180)
r0_composite = D / Dr0_composite
heights = [12.5] # [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0]# [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
r0 = (r0_composite / nlayers^(-3/5)) .* ones(nlayers)  # m
seeds = [713]# [713, 1212, 525118]
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    l0=l0,
    L0=L0,
    r0=r0, 
>>>>>>> main
    wind=wind, 
    heights=heights, 
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
<<<<<<< HEAD
    propagate=propagate,
=======
>>>>>>> main
    seeds=seeds, 
    FTYPE=FTYPE,
    verb=verb
)
########## Create phase screens ###########
<<<<<<< HEAD
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
header = create_header(λ, units="rad")
writefits(atmosphere.phase, "$(folder)/Dr0_$(round(Int64, Dr0_ref_vertical))_phase_full$(id).fits", header=header)
=======
calculate_screen_size!(atmosphere, observations_full, object, patches, verb=verb)
calculate_pupil_positions!(atmosphere, observations_full, verb=verb)
calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full, verb=verb)
create_phase_screens!(atmosphere, observations_full)
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full, 0.1)
atmosphere.opd .*= atmosphere.masks
# opd_smooth .*= atmosphere.masks

# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, build_dim=image_dim, propagate=propagate)
# writefits(atmosphere.masks, "$(folder)/layer_masks$(id).fits", header=header)
# writefits(observations_full.A, "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_amplitude$(id).fits")
# writefits(observations_wfs.A, "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_amplitude$(id).fits")
# writefits(atmosphere.ϕ, "$(folder)/Dr0_$(Dr0_composite)_phase_full$(id).fits", header=header)
# writefits(patches.ϕ_slices, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_slices$(id).fits", header=header)
# writefits(patches.ϕ_composite, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_composite$(id).fits", header=header)
# writefits(atmosphere.opd, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full$(id).fits")
>>>>>>> main
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
# exit()
###########################################

########## Create Full-Ap images ##########
<<<<<<< HEAD
# using BenchmarkTools
# @btime [create_detector_images($patches, $observations[dd], $atmosphere, $object, build_dim=$image_dim, noise=$noise, verb=$verb) for dd=1:length(observations)]
[create_detector_images(patches, observations[dd], atmosphere, object, build_dim=image_dim, noise=noise, verb=verb) for dd=1:length(observations)]
[writefits(observations[dd].images, "$(folder)/Dr0_$(round(Int64, Dr0_ref_vertical))_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_images$(id).fits", header=create_header(observations[dd]), times=observations[dd].times) for dd=1:length(observations)]
=======
create_images_eff(patches, observations, atmosphere, masks, object, build_dim=image_dim, noise=noise)
outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_images$(id).fits"
writefits(observations_full.images, outfile, header=header)
outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH$(nsubaps_side)x$(nsubaps_side)_images$(id).fits"
writefits(observations_wfs.images, outfile, header=header)
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_monochromatic_images$(id).fits"
# writefits(observations_full.monochromatic_images, outfile, header=header)
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_psfs$(id).fits"
# writefits(patches.psfs[1], outfile, header=header)
>>>>>>> main
###########################################

## Isoplanatic
# Naive threadix: 1.278 s (546486 allocations: 137.16 MiB)
# OhMyThreads.Channel: 1.264 s (533701 allocations: 144.43 MiB)
#                      1.240 s (536749 allocations: 144.49 MiB)
## Anisoplanatic
# Naive threadix: 197.684 s (751385 allocations: 144.23 MiB)
# OhMyThreads.Channel: 205.700 s (748062 allocations: 151.27 MiB)
#                      102.606 s (743478 allocations: 151.16 MiB)      
