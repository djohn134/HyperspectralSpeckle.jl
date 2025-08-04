include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data"
id = "_test"
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64
nsubaps_side = 6
nλ = 1
nλint = 1
λ_ref = 500.0e-9
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
resolution = mean(λ) / Δλ
###########################################

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = true
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
D_inner_frac = 0.0
aperture_area = pi * (D / 2)^2 * (1 - D_inner_frac^2)
fov = 10.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
DTYPE = UInt16
saturation = 30000.0  # e⁻
gain = saturation / (typemax(DTYPE))  # e⁻ / ADU
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
datafile = "$(folder)/Dr0_20_ISH1x1_images_test.fits"
images_full, ~, nepochs, image_dim, exptime_full, times_full = readimages(datafile, FTYPE=FTYPE)
detector_full = Detector(
    qe=qe,
    rn=rn,
    gain=gain,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime_full,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
observations_full = Observations(
    times_full,
    images_full,
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    area=aperture_area,
    ϕ_static=ϕ_static_full,
    verb=verb,
    FTYPE=FTYPE
)
# observations = [observations_full]
datafile = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images_test.fits"
images_wfs, nsubaps, ~, wfs_dim, exptime_wfs, times_wfs = readimages(datafile, FTYPE=FTYPE)
optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    gain=gain,
    pixscale=pixscale_wfs,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime_wfs,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
observations_wfs = Observations(
    times_wfs,
    images_wfs,
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    area=aperture_area,
    nsubaps_side=nsubaps_side,
    ϕ_static=ϕ_static_wfs,
    verb=verb,
    FTYPE=FTYPE
)
observations = [observations_wfs, observations_full]
background = mean(fit_background(observations_full))
###########################################

########## Create Full-Ap Masks ###########
masks_full = Masks(
    image_dim,
    λ,
    nsubaps_side=1, 
    D_inner_frac=D_inner_frac,  
    λ_nyquist=λ_nyquist, 
    verb=verb,
    FTYPE=FTYPE
)
# masks = [masks_full]
masks_wfs = Masks(
    image_dim,
    λ,
    nsubaps_side=nsubaps_side,
    D_inner_frac=D_inner_frac, 
    λ_nyquist=λ_nyquist, 
    verb=verb,
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
masks = [masks_wfs, masks_full]
###########################################

############ Object Parameters ############
object_range = 500.0e3  # km
############## Create object ##############
object_arr = max.(dropdims(mean(observations_full.images, dims=(3, 4)), dims=(3, 4)) .- background/observations_full.dim^2, 0)
object_arr = repeat(object_arr, 1, 1, nλ)
object_arr ./= sum(object_arr)
object_arr .*= (mean(sum(observations_full.images, dims=(1, 2, 3))) - background)
# object_arr = readfits("$(folder)/object_recon.fits")
~, spectrum = solar_spectrum(λ=λ)
object = Object(
    object_arr,
    λ=λ,
    object_range=object_range, 
    fov=fov,
    dim=observations_full.dim,
    background=background,
    spectrum=spectrum,
    scaled=true,
    FTYPE=FTYPE,
    verb=verb
)
###########################################

########## Atmosphere Parameters ##########
heights = [0.0, 7000.0, 12500.0]
wind_speed = wind_profile_roberts2011(heights, ζ)
heights .*= 0.0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    masks,
    object, 
    patches,
    wind=wind, 
    heights=heights,
    transmission=transmission,
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    create_screens=false,
    FTYPE=FTYPE,
    verb=verb
)
######### Set phase screens start #########
atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
# atmosphere.phase = readfits("$(folder)/Dr0_40_phase_full.fits")
###########################################
object.object .*= observations_full.detector.gain / (observations_full.detector.exptime * observations_full.aperture_area * mean(observations_full.optics.response) * mean(atmosphere.transmission))
object.background *= observations_full.detector.gain / (observations_full.detector.exptime * observations_full.aperture_area * mean(observations_full.optics.response) * mean(atmosphere.transmission))

######### Reconstruction Object ###########
reconstruction = Reconstruction(
    atmosphere,
    observations,
    object,
    patches,
    λmin=λmin,
    λmax=λmax,
    nλ=nλ,
    nλint=nλint,
    niter_mfbd=10,
    maxiter=10,
    # indx_boot=[1:2],
    wavefront_parameter=:phase,
    minimization_scheme=:mle,
    noise_model=:gaussian,
    maxeval=Dict("wf"=>10000, "object"=>10000),
    smoothing=true,
    # fwhm_schedule=ConstantSchedule(0.5),
    build_dim=image_dim,
    verb=verb,
    plot=plot,
    FTYPE=FTYPE
);
reconstruct!(reconstruction, observations, atmosphere, object, masks, patches, write=true, folder=folder, id=id)
###########################################
