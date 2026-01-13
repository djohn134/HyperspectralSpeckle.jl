include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "test"
id = "_aniso"
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
isoplanatic = false
patch_dim = 128
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
D_inner_frac = 0.0
fov = 30.0
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
# # ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
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
# observations = [observations_wfs]
[observations[dd].phase = readfits("test/wf_phase.fits") for dd=1:length(observations)]
background = 0*observations_full.dim^2  # mean(fit_background(observations_full))
###########################################


############ Object Parameters ############
object_range = 500.0e3  # km
############## Create object ##############
# object_arr = max.(dropdims(mean(observations_full.images, dims=(3, 4)), dims=(3, 4)) .- background/observations_full.dim^2, 0)
# object_arr = repeat(object_arr, 1, 1, nλ)
# object_arr ./= sum(object_arr)
# object_arr .*= (mean(sum(observations_full.images, dims=(1, 2, 3))) - background)
object_arr = readfits("$(folder)/object_recon_iso.fits", FTYPE=FTYPE)
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
heights = [0.0, 7000.0, 12500.0]
wind_speed = wind_profile_roberts2011(heights, ζ)
heights .*= 0.0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("/home/dan/Desktop/HyperspectralSpeckle.jl/data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
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
# atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, nλ)
atmosphere.phase = readfits("$(folder)/phase_recon_iso.fits", FTYPE=FTYPE)
# atmosphere.opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
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
    niter_mfbd=1,
    maxiter=1000,
    # indx_boot=[1:2],
    wavefront_parameter=:phase,
    frozen_flow=true,
    minimization_scheme=:mle,
    noise_model=:mixed,
    maxeval=Dict("wf"=>1, "object"=>1000),
    grtol=5e-3,
    frtol=1e-4,
    xrtol=1e-4,
    smoothing=false,
    # fwhm_schedule=ConstantSchedule(1.0),
    build_dim=image_dim,
    mfbd_verb_level=:full,
    verb=verb,
    plot=plot,
    FTYPE=FTYPE
);
###########################################

hmin = [5000.0, 10000.0]  # m
hmax = [10000.0, 15000.0]  # m
hstep = [1000.0, 1000.0]  # m
niters = 3
ϵ_heights, height_trials = height_solve!(observations, atmosphere, object, patches, reconstruction, hmin=hmin, hmax=hmax, hstep=hstep, niters=niters)
writefile(cat(height_trials, dims=1), cat(ϵ_heights, dims=1), "$(folder)/heights$(id).dat")

using GLMakie
fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], xlabel="Height [m]", ylabel="ϵ")
for l=1:atmosphere.nlayers-1
    lines!(ax, height_trials[l], ϵ_heights[l])
end
save("$(folder)/heights$(id).png", fig, px_per_unit=16)

## Write anisoplanatic phases and images ##
writefits(object.object, "$(folder)/object_recon$(id).fits")
writefits(atmosphere.phase, "$(folder)/phase_recon$(id).fits")
writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
res_full = observations_full.model_images .- observations_full.images
writefits(res_full, "$(folder)/residuals_ISH1x1_recon$(id).fits")
# writefile(reconstruction.ϵ, "$(folder)/recon$(id).dat")
###########################################
