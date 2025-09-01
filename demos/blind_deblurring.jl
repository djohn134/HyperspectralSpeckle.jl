include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "test"
id = "_onlyFullAp_mixed"
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
nsubaps_side = 6
datafile_wfs = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images.fits"
images_wfs, nsubaps, ~, wfs_dim, exptime_wfs, times_wfs = readimages(datafile_wfs, FTYPE=FTYPE)
datafile_full = "$(folder)/Dr0_20_ISH1x1_images.fits"
images_full, ~, ~, image_dim, exptime_full, times_full = readimages(datafile_full, FTYPE=FTYPE)
nλ = 1
nλint = 1
λ_ref = 500.0e-9  # m
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9  # m
λ = collect(range(λmin, stop=λmax, length=nλ))  # m
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)  # m
resolution = mean(λ) / Δλ
###########################################

########## Anisopatch Parameters ##########
isoplanatic = true
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
D_inner_frac = 0.0
fov = 10.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
DTYPE = UInt16
saturation = 30000.0  # e⁻
gain = saturation / (typemax(DTYPE))  # e⁻ / ADU
qefile = "../data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    gain=gain,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    label="Full Aperture",
    FTYPE=FTYPE
)
# diversity = Diversity(
#     4,
#     1.0,
#     2*(maximum(times_full)-minimum(times_full)),
#     minimum(times_full),
#     verb=verb,
#     FTYPE=FTYPE
# )
### Create Full-Ap Observations object ####
# ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
observations_full = Observations(
    times_full,
    images_full,
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    D_inner_frac=D_inner_frac,
    # ϕ_static=ϕ_static_full,
    # diversity=diversity,
    verb=verb,
    label="Full Aperture",
    FTYPE=FTYPE
)
observations = [observations_full]
# optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# detector_wfs = Detector(
#     qe=qe,
#     rn=rn,
#     gain=gain,
#     pixscale=pixscale_wfs,
#     λ=λ,
#     λ_nyquist=λ_nyquist,
#     exptime=exptime,
#     verb=verb,
#     label="Wavefront Sensor",
#     FTYPE=FTYPE
# )
# ### Create Full-Ap Observations object ####
# ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
# observations_wfs = Observations(
#     times_wfs,
#     images_wfs,
#     optics_wfs,
#     detector_wfs,
#     ζ=ζ,
#     D=D,
#     D_inner_frac=D_inner_frac,
#     nsubaps_side=nsubaps_side,
#     # ϕ_static=ϕ_static_wfs,
#     build_dim=observations_full.dim,
#     verb=verb,
#     label="Wavefront Sensor",
#     FTYPE=FTYPE
# )
# nsubaps = observations_wfs.masks.nsubaps
# observations_wfs.masks.scale_psfs = observations_full.masks.scale_psfs
# observations = [observations_wfs, observations_full]
[observations[dd].phase = readfits("test/phase_ISH1x1_recon_onlyFullAp.fits") for dd=1:length(observations)]
background = 0  # mean(fit_background(observations_full))
###########################################


############ Object Parameters ############
object_range = 500.0e3  # km
############## Create object ##############
# object_arr = max.(dropdims(mean(observations_full.images, dims=(3, 4)), dims=(3, 4)) .- background/observations_full.dim^2, 0)
# object_arr = repeat(object_arr, 1, 1, nλ)
# object_arr ./= sum(object_arr)
# object_arr .*= (mean(sum(observations_full.images, dims=(1, 2, 3))) - background)
object_arr = readfits("$(folder)/object_recon_onlyFullAp.fits")
~, spectrum = solar_spectrum(λ=λ)
object = Object(
    object_arr,
    λ=λ,
    nλint=nλint,
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
# heights = [0.0, 7000.0, 12500.0]
# wind_speed = wind_profile_roberts2011(heights, ζ)
# # heights .*= 0.0
# wind_direction = [45.0, 125.0, 135.0]
# wind = [wind_speed wind_direction]
# nlayers = length(heights)
# sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
# sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("/home/dan/Desktop/HyperspectralSpeckle.jl/data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    object, 
    patches,
    # wind=wind, 
    # heights=heights,
    transmission=transmission,
    # sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    # sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    create_screens=false,
    FTYPE=FTYPE,
    verb=verb
)
######### Set phase screens start #########
# atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
# atmosphere.phase = readfits("test/phase_ISH1x1_recon.fits")
###########################################
# object.object .*= observations_full.detector.gain / (observations_full.detector.exptime * observations_full.aperture_area * mean(observations_full.optics.response) * mean(atmosphere.transmission))
# object.background *= observations_full.detector.gain / (observations_full.detector.exptime * observations_full.aperture_area * mean(observations_full.optics.response) * mean(atmosphere.transmission))

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
    wavefront_parameter=:opd,
    frozen_flow=false,
    minimization_scheme=:mle,
    noise_model=:mixed,
    maxeval=Dict("wf"=>100, "object"=>100),
    smoothing=false,
    # fwhm_schedule=ConstantSchedule(0.5),
    build_dim=image_dim,
    verb=verb,
    plot=plot,
    FTYPE=FTYPE
);
reconstruct!(reconstruction, observations, atmosphere, object, patches, write=true, folder=folder, id=id)
###########################################
