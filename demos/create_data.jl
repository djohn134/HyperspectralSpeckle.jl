include("../src/HyperspectralSpeckle.jl")
using Main.HyperspectralSpeckle
using Statistics


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data/test"
id = ""
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64

nsubaps_side = 6
nλ = 1
λ_ref = 500.0e-9  # m
λmin = 500.0e-9  # m
λmax = 500.0e-9  # m
λ = collect(range(λmin, stop=λmax, length=nλ))  # m
λ_nyquist = mean(λ)  # m
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)  # m
resolution = mean(λ) / Δλ

########## Create Full-Ap Masks ###########
masks_full = Masks(
    image_dim,
    λ,
    nsubaps_side=1,  
    λ_nyquist=λ_nyquist, 
    verb=verb,
    FTYPE=FTYPE
)
# masks = [masks_full]
masks_wfs = Masks(
    image_dim,
    λ,
    nsubaps_side=nsubaps_side,
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
masks = [masks_full, masks_wfs]
header = create_header(λ, "unitless")
[writefits(masks[dd].masks, "$(folder)/masks_ISH$(masks[dd].nsubaps_side)x$(masks[dd].nsubaps_side)$(id).fits", header=header) for dd=1:length(masks)]
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 50.0  # arcsec
pixscale_full = fov / image_dim  # arcsec/pix
pixscale_wfs = pixscale_full .* nsubaps_side  # arcsec/pix
DTYPE = UInt16
# DTYPE = FTYPE
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)  # e⁻ / ph
# qe = ones(FTYPE, nλ)  # e⁻ / ph
rn = 2.0  # e⁻
saturation = 30000.0  # e⁻
# saturation = 1e99  # e⁻
gain = saturation / (typemax(DTYPE))  # e⁻ / ADU
# gain = 1.0  # e⁻ / ADU
exptime = 5e-3  # sec
nepochs = 5
times = collect(0:nepochs-1) .* exptime
noise = true
ζ = 0.0  # deg
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
# filter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[1.0, 1.0], FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# optics_full = OpticalSystem([filter], λ, verb=verb, FTYPE=FTYPE)
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
    DTYPE=DTYPE
)
### Create Full-Ap Observations object ####
ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
# ϕ_static_full = repeat(masks_full.masks[:, :, 1, 1] .* create_zernike_screen(image_dim, image_dim÷4, 4, 4.0, FTYPE=FTYPE), 1, 1, nλ)
# writefits(ϕ_static_full, "$(folder)/defocus4.fits")
observations_full = Observations(
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    times=times,
    nsubaps=1,
    dim=image_dim,
    ϕ_static=ϕ_static_full,
    verb=verb,
    FTYPE=FTYPE
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
    DTYPE=DTYPE
)
### Create Full-Ap Observations object ####
ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
observations_wfs = Observations(
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    times=times,
    nsubaps=masks_wfs.nsubaps,
    nsubaps_side=nsubaps_side,
    dim=wfs_dim,
    ϕ_static=ϕ_static_wfs,
    verb=verb,
    FTYPE=FTYPE
)
observations = [observations_full, observations_wfs]
###########################################

############ Object Parameters ############
objectfile = "data/OCNR2.fits"
# object_arr, ~ = template2object(objectfile, image_dim, λ, FTYPE=FTYPE)
object_arr = repeat(block_reduce(readfits(objectfile, FTYPE=FTYPE), image_dim), 1, 1, nλ)
~, spectrum = solar_spectrum(λ=λ)
mag = 4.0
background_mag = Inf  # mag / arcsec^2
flux = mag2flux(mag, filter, D=D, ζ=ζ, exptime=exptime)  # ph
background_flux = mag2flux(background_mag, filter, D=D, ζ=ζ, exptime=exptime)  # ph / arcsec^2
background_flux *= fov^2  # ph
object_range = 300.0e3  # m
############## Create object ##############
object = Object(
    object_arr,
    flux=flux,
    background_flux=background_flux,
    λ=λ,
    fov=fov,
    object_range=object_range, 
    dim=image_dim,
    spectrum=spectrum,
    verb=verb,
    FTYPE=FTYPE
)
header = create_header(λ, "ph/m")
writefits(object.object, "$(folder)/object_truth_spectral$(id).fits", header=header)
header = create_header(λ, "ph")
writefits(dropdims(sum(object.object, dims=3), dims=3)*Δλ, "$(folder)/object_truth$(id).fits", header=header)
###########################################

########## Anisopatch Parameters ##########
isoplanatic = false
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

########## Atmosphere Parameters ##########
l0 = 0.01  # m
L0 = 100.0  # m
Dr0_ref_vertical = 20.0
Dr0_ref_composite = Dr0_ref_vertical * secd(ζ)
r0_ref_composite = D / Dr0_ref_composite
heights = [0.0, 7000.0, 12500.0]  # m
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0, 125.0, 135.0]  # deg
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
r0_ref = composite_r0_to_layers(r0_ref_composite, heights, λ_ref, ζ)
seeds = [713, 1212, 525118]
Dmeta = D .+ (fov/206265) .* heights
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
# transmission = ones(FTYPE, nλ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    masks,
    object, 
    patches,
    l0=l0,
    L0=L0,
    r0=r0_ref, 
    wind=wind, 
    heights=heights, 
    transmission=transmission,
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    propagate=propagate,
    seeds=seeds, 
    FTYPE=FTYPE
)
########## Create phase screens ###########
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
header = create_header(λ, "rad")
writefits(atmosphere.phase, "$(folder)/Dr0_$(round(Int64, Dr0_ref_composite))_phase_full$(id).fits", header=header)
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
###########################################

########## Create Full-Ap images ##########
[create_images(patches, observations[dd], atmosphere, masks[dd], object, build_dim=image_dim, noise=noise) for dd=1:length(observations)]
header = create_header(λ, "counts")
[writefits(observations[dd].images, "$(folder)/Dr0_$(round(Int64, Dr0_ref_composite))_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_images$(id).fits", header=header) for dd=1:length(observations)]
###########################################
