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
nepochs = 33
nλ = 1
λ_ref = 500.0
λmin = λ_nyquist = 500.0
λmax = 500.0
λ = collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
resolution = mean(λ) / Δλ

header = (
    ["WAVELENGTH_START", "WAVELENGTH_END", "WAVELENGTH_STEPS"], 
    [λmin, λmax, nλ], 
    ["Shortest wavelength of mask [nm]", "Largest wavelength of mask [nm]", "Number of wavelength steps"]
)
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
[writefits(masks[dd].masks, "$(folder)/masks_ISH$(masks[dd].nsubaps_side)x$(masks[dd].nsubaps_side)$(id).fits", header=header) for dd=1:length(masks)]
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 5.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
noise = true
ζ = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
# filter = OpticalElement(λ=[0.0, 10000.0], response=[1.0, 1.0], FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[0.0, 10000.0], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# optics_full = OpticalSystem([filterV], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE
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
    nepochs=nepochs,
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
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
observations_wfs = Observations(
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
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
# object, ~ = template2object(objectfile, dim, λ, FTYPE=FTYPE)
object_arr = repeat(block_reduce(readfits(objectfile, FTYPE=FTYPE), image_dim), 1, 1, nλ)
~, spectrum = solar_spectrum(λ=λ)
mag = 6.0
background_mag = Inf
flux = mag2flux(λ, spectrum, mag, filter, D=D, ζ=ζ, exptime=exptime)
background_flux = mag2flux(λ, ones(nλ), background_mag, filter, D=D, ζ=ζ, exptime=exptime)
background_flux *= fov^2
object_height = 515.0  # km
############## Create object ##############
object = Object(
    object_arr,
    flux=flux,
    background_flux=background_flux,
    λ=λ,
    fov=fov,
    height=object_height, 
    dim=image_dim,
    spectrum=spectrum,
    verb=verb,
    FTYPE=FTYPE
)
writefits(object.object, "$(folder)/object_truth$(id).fits", header=header)
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
Dr0_ref_composite = Dr0_ref_vertical * sec(ζ*pi/180)
r0_ref_composite = D / Dr0_ref_composite
heights = [0.0, 7.0, 12.5]
# heights = [0.0]
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0, 125.0, 135.0]
# wind_direction = [45.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
r0_ref = composite_r0_to_layers(r0_ref_composite, heights, λ_ref, ζ)
seeds = [713, 1212, 525118]
# seeds = [713]
Dmeta = D .+ (fov/206265) .* (heights .* 1000)
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
# ~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
transmission = ones(FTYPE, nλ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations_full, 
    masks_full,
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

writefits(atmosphere.masks, "$(folder)/layer_masks.fits", header=header)
writefits(atmosphere.phase, "$(folder)/Dr0_$(round(Int64, Dr0_ref_composite))_phase_full$(id).fits", header=header)
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
###########################################

########## Create Full-Ap images ##########
create_images(patches, observations, atmosphere, masks, object, build_dim=image_dim, noise=noise)
[writefits(observations[dd].images, "$(folder)/Dr0_$(round(Int64, Dr0_ref_composite))_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_images$(id).fits", header=header) for dd=1:length(observations)]
###########################################
