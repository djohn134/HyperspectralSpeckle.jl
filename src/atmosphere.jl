using Random
using Statistics
using LazyAlgebra
# using FourierTools
using TwoDimensional
using LinearInterpolators 
import Interpolations: interpolate, Gridded, Linear


mutable struct Atmosphere{T<:AbstractFloat}
    nlayers::Int64
    l0::T
    L0::T
    r0::Vector{T}
    wind::Matrix{T}
    heights::Vector{T}
    transmission::Vector{T}
    sampling_nyquist_mperpix::Vector{T}
    sampling_nyquist_arcsecperpix::Vector{T}
    λ::Vector{T}
    λ_nyquist::T
    λ_ref::T
    nλ::Int64
    Δλ::T
    propagate::Bool
    common_opd::Bool
    seeds::Vector{Int64}
    masks::Array{T, 4}
    dim::Int64
    positions::Array{T, 4}
    A::Array{T, 4}
    opd::Array{T, 3}
    phase::Array{T, 4}
    phase_slices::Array{T, 5}
    phase_composite::Array{T, 4}
    function Atmosphere(; 
            l0=Inf,
            L0=Inf,
            r0=[Inf],
            wind=[Inf], 
            heights=[Inf],
            transmission=[1.0],
            sampling_nyquist_mperpix=[Inf],
            sampling_nyquist_arcsecperpix=[Inf],
            λ=[Inf], 
            λ_nyquist=Inf,
            λ_ref=Inf,
            common_opd=true,
            propagate=false,
            seeds=[0],
            verb=true,
            FTYPE=Float64
        )
        nlayers = length(heights)
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)        
        
        if verb == true
            print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Atmosphere\n"); print(Crayon(reset=true))
            println("\tNumber of layers: $(nlayers) layers")
            println("\tWind speed: $(wind[:, 1]) m/s")
            println("\tWind direction: $(wind[:, 2]) deg")
            println("\tInner scale: $(l0) m")
            println("\tOuter scale: $(L0) m")
            println("\tFried paremeter: $(r0) m")
            println("\tReference wavelength: $(λ_ref) nm")
            println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
            println("\tNumber of wavelengths: $(length(λ)) wavelengths")
            println("\tPropagate: $(propagate)")
            println("\tCommon opd: $(common_opd)")
        end

        return new{FTYPE}(nlayers, l0, L0, r0, wind, heights, transmission, sampling_nyquist_mperpix, sampling_nyquist_arcsecperpix, λ, λ_nyquist, λ_ref, nλ, Δλ, propagate, common_opd, seeds)
    end
end

function layer_nyquist_sampling_arcsecperpix(D, fov, layer_heights, image_dim)
    Dmeta = D .+ (fov / 206265) .* (layer_heights .* 1000)
    sampling_nyquist_arcsecperpix = (fov / image_dim) .* (2*D ./ Dmeta)
    return sampling_nyquist_arcsecperpix
end

function layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
    sampling_nyquist_mperpix = (2 * D / image_dim) .* ones(nlayers)
    return sampling_nyquist_mperpix
end

@views function generate_phase_screen_mvK(r0, dim, sampling, L0, l0; seed=0, FTYPE=Float64)
    Random.seed!(seed)
    # Function for creating a random draw phase screen
    # Creates a phase screen based on the FT method
    # Setup the PSD
    del_f = 1 / (dim*sampling);		    # Frequency grid space [1/m]
    # Fequency grid [1/m]
    fx = del_f .* ([j for i=1:dim, j=1:dim] .- (dim÷2+1))
    fy = fx'
    f = hypot.(fx, fy);
    fm = FTYPE(5.92 / l0 / 2pi);		# Inner scale frequency		# Use of 'π'
    f0 = FTYPE(1 / L0);          # Outer scaling frequency
    # Modified von Karman atmospheric phase PSD
    PSDᵩ = FTYPE(0.023) * r0^(-5/3) .* exp.(-(f ./ fm).^2) ./ (f.^2 .+ f0^2).^(11/6);
    PSDᵩ[dim÷2+1, dim÷2+1] = 0;
    # Random draws of Fourier coefficients
    cn = randn(Complex{FTYPE}, dim, dim) .* sqrt.(PSDᵩ) .* del_f;
    # Synthesize the phase screen
    ϕ = real.(ift(cn)) .* FTYPE(dim^2)
    return FTYPE.(ϕ)
end

@views function generate_phase_screen_subharmonics(r0, dim, sampling, L0, l0; seed=0, FTYPE=Float64)
    # Following Schmidt Listing 9.3, page 170
    # Augments the above phase screen method to include subharmonics to improve the low spatial frquencies.
    D = dim * sampling
    ϕ_hi = generate_phase_screen_mvK(r0, dim, sampling, L0, l0, seed=seed, FTYPE=FTYPE); 	 # High frequency screen from FFT method
    x = sampling .* ([j for i=1:dim, j=1:dim] .- (dim÷2+1))
    y = x'
    
    ϕ_lo = zeros(size(ϕ_hi)); 	 # Initialise low-freq screen,
    # loop over frequency grids with spacing 1/(3^p*L)
    for p = 1:3
        del_f = 1 / (3^p*D);		# Frequency grid spacing [1/m]
        fx = del_f .* ([j for i=-1:1, j=-1:1])
        fy = fx'
        f  = hypot.(fx, fy)
        fm = FTYPE(5.92 / l0 / 2pi);		# Inner scale frequency [1/m]  # 'π' may not work - Matt
        f0 = FTYPE(1 / L0); 			    # Outer scale frequency [1/m]
        PSDᵩ = FTYPE(0.023) * r0^(-5/3) .* exp.(-(f ./ fm).^2) ./ (f.^2 .+ f0^2).^(11/6); 	     # modified von Karmen atmospheric phase PSD
        PSDᵩ[2, 2] = 0;
        cn = randn(Complex{FTYPE}, 3) .* sqrt.(PSDᵩ) .* del_f; 	     # Random draws of Fourier coefficients
        SH = zeros(FTYPE, dim, dim);
        # Loop over frequencies on this grid
        for ii = 1:9
             SH .+= real.(cn[ii] .* cis.(2pi.*(fx[ii].*x + fy[ii].*y))); 
        end
        ϕ_lo .+= SH;   # Accumulate subharmonics
    end

    ϕ_lo .-= mean(ϕ_lo);
    ϕ = ϕ_lo .+ ϕ_hi
    return FTYPE.(ϕ)
end

function generate_phase_screen_kolmogorov(atmosphere, observations; grow=4, FTYPE=Float64)
    Random.seed!(atmosphere.seeds[l])

    big_mask = zeros(FTYPE, atmosphere.dim, atmosphere.dim)
    reference_mask = make_ish_masks(observations.dim, 1, atmosphere.λ_ref, λ_nyquist=atmosphere.λ_nyquist)
    center_ixs = atmosphere.dim÷2-observations.dim÷2:atmosphere.dim÷2+observations.dim÷2-1
    big_mask[center_ixs, center_ixs] .= reference_mask[:, :, end, 1]    

    x = [j for i=1:grow*atmosphere.dim, j=1:grow*atmosphere.dim] .- (grow*atmosphere.dim÷2+1);
    rr = hypot.(x, x')

    c = (grow*atmosphere.dim)÷2 + 1
    PSDᵩ = FTYPE(0.023) * (2*Dr0[l])^(5/3) .* rr.^(-11/6)
    PSDᵩ[c, c] = 0
    PSDᵩ .*= randn(Complex{FTYPE}, Int(grow*atmosphere.dim), Int(grow*atmosphere.dim))
    ϕ = (real.(ift(PSDᵩ) .* FTYPE(grow*atmosphere.dim)))[c-atmosphere.dim÷2:c+atmosphere.dim÷2-1, c-atmosphere.dim÷2:c+atmosphere.dim÷2-1]
    ϕ .*= (Dr0[l] / (var(ϕ[big_mask .> 0, l])/FTYPE(1.03))^(3/5))^(5/6)
    return FTYPE.(ϕ)
end

function calculate_screen_size!(atmosphere, observations; verb=true)
    Δpos_meters = observations.detector.exptime .* [atmosphere.wind[:, 1].*sin.(atmosphere.wind[:, 2].*pi/180) atmosphere.wind[:, 1].*cos.(atmosphere.wind[:, 2].*pi/180)]'
    Δpos_pix = Δpos_meters ./ minimum(atmosphere.sampling_nyquist_mperpix)
    Δpos_pix_total = Δpos_pix .* (observations.nepochs - 1)
    Δpix_refraction = maximum(abs.(refraction_at_layer_pix(atmosphere, observations)))
    atmosphere.dim = nextprod((2, 3, 5, 7), maximum(ceil.(abs.(Δpos_pix_total))) + observations.dim + Δpix_refraction)
    if verb == true
        println("\tSize: $(atmosphere.dim)×$(atmosphere.dim) pixels")
    end
end

function calculate_screen_size!(atmosphere, observations, object, patches; verb=true)
    Δpos_meters = observations.detector.exptime .* [atmosphere.wind[:, 1].*sin.(atmosphere.wind[:, 2].*pi/180) atmosphere.wind[:, 1].*cos.(atmosphere.wind[:, 2].*pi/180)]'
    Δpos_pix = Δpos_meters ./ minimum(atmosphere.sampling_nyquist_mperpix)
    Δpos_pix_total = Δpos_pix .* (observations.nepochs - 1)
    Δpix_refraction = maximum(abs.(refraction_at_layer_pix(atmosphere, observations)))
    Δpix_aniso = 2*maximum(abs.(patches.positions)) * (object.sampling_arcsecperpix / 206265) * maximum(atmosphere.heights)*1000 / minimum(atmosphere.sampling_nyquist_mperpix)
    atmosphere.dim = nextprod((2, 3, 5, 7), maximum(ceil.(abs.(Δpos_pix_total))) + observations.dim + Δpix_refraction + Δpix_aniso)
    if verb == true
        println("\tSize: $(atmosphere.dim)×$(atmosphere.dim) pixels")
    end
end

@views function create_phase_screens!(atmosphere, observations; verb=true)
    FTYPE = gettype(observations)

    if verb == true
        println("Creating $(atmosphere.nlayers) layers of size $(atmosphere.dim)×$(atmosphere.dim) with r0=$(atmosphere.r0) m (at $(atmosphere.λ_ref) nm)")
    end

    atmosphere.A = ones(FTYPE, observations.dim, observations.dim, observations.nepochs, atmosphere.nλ)
    atmosphere.opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
    atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
    
    if atmosphere.common_opd == true
        ϕ_ref = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
        sampling_ref = atmosphere.sampling_nyquist_mperpix .* (atmosphere.λ_ref/atmosphere.λ_nyquist)        
        Threads.@threads for l=1:atmosphere.nlayers
            ϕ_ref[:, :, l] .= generate_phase_screen_subharmonics(atmosphere.r0[l], atmosphere.dim, sampling_ref[l], atmosphere.L0, atmosphere.l0, seed=atmosphere.seeds[l], FTYPE=FTYPE)
            atmosphere.opd[:, :, l] .= ϕ_ref[:, :, l] .* atmosphere.λ_ref / FTYPE(2pi)
            atmosphere.opd[:, :, l] .-= mean(atmosphere.opd[findall(atmosphere.masks[:, :, l, 1] .> 0), l])
            # atmosphere.opd[:, :, l] .-= fit_plane(atmosphere.opd[:, :, l], atmosphere.masks[:, :, l, 1])
            for w=1:atmosphere.nλ
                atmosphere.phase[:, :, l, w] .= FTYPE(2pi) / atmosphere.λ[w] .* atmosphere.opd[:, :, l]
            end
        end
    else
        Threads.@threads for w=1:atmosphere.nλ
            sampling_mperpix = atmosphere.sampling_nyquist_mperpix .* (atmosphere.λ[w]/atmosphere.λ_nyquist)
            r0λ = atmosphere.r0 * (atmosphere.λ[w] / atmosphere.λ_ref)^(6/5)
            for l=1:atmosphere.nlayers
                atmosphere.phase[:, :, l, w] .= generate_phase_screen_subharmonics(r0λ[l], atmosphere.dim, sampling_mperpix[l], atmosphere.L0, atmosphere.l0, seed=atmosphere.seeds[l], FTYPE=FTYPE)
                atmosphere.phase[:, :, l, w] .-= mean(atmosphere.phase[findall(atmosphere.masks[:, :, l, w] .> 0), l, w])
            end
        end
    end
    # atmosphere.opd .*= atmosphere.masks[:, :, :, 1]
end

@views function calculate_pupil_positions!(atmosphere, observations; verb=true)
    if verb == true
        println("Calculating pupil positions for $(atmosphere.nlayers) layers at $(observations.nepochs) times and $(atmosphere.nλ) wavelengths")
    end
    
    FTYPE = gettype(atmosphere)
    Δpos_meters = observations.detector.exptime .* [atmosphere.wind[:, 1].*sin.(atmosphere.wind[:, 2].*pi/180) atmosphere.wind[:, 1].*cos.(atmosphere.wind[:, 2].*pi/180)]'
    Δpos_pix = Δpos_meters ./ repeat(atmosphere.sampling_nyquist_mperpix', 2, 1)
    Δpos_pix_total = Δpos_pix .* (observations.nepochs - 1)
    atmosphere.positions = zeros(FTYPE, 2, observations.nepochs, atmosphere.nlayers, atmosphere.nλ)
    Δpix_refraction = refraction_at_layer_pix(atmosphere, observations)
    for w=1:atmosphere.nλ
        for l=1:atmosphere.nlayers
            atmosphere.positions[:, 1, l, w] .= (atmosphere.dim .- Δpos_pix_total[:, l]) / 2
            atmosphere.positions[1, 1, l, w] -= Δpix_refraction[l, w]
            for t=2:observations.nepochs
                atmosphere.positions[:, t, l, w] .= atmosphere.positions[:, 1, l, w] .+ (Δpos_pix[:, l] .* (t-1))
            end
        end
    end
end

function layer_scale_factors(layer_heights, object_height)
    return 1 .- layer_heights ./ object_height
end

function air_refractive_index_minus_one(λ; pressure=69.328, temperature=293.15, H2O_pressure=1.067)
    """
    Adapted from GalSim to take nm 
    Return the refractive index of air as function of wavelength.

    Uses the formulae given in Filippenko (1982), which appear to come from Edlen (1953),
    and Coleman, Bozman, and Meggers (1960).  The units of the original formula are non-SI,
    being mmHg for pressure (and water vapor pressure), and degrees C for temperature.  This
    function accepts SI units, however, and transforms them when plugging into the formula.

    The default values for temperature, pressure and water vapor pressure are expected to be
    appropriate for LSST at Cerro Pachon, Chile, but they are broadly reasonable for most
    observatories.

    Parameters:
        wave:             Wavelength array in nm
        pressure:         Air pressure in kiloPascals.
        temperature:      Temperature in Kelvins.
        H2O_pressure:     Water vapor pressure in kiloPascals.

    Returns:
        the refractive index minus 1.
    """
    P = pressure * 7.50061683 # kPa -> mmHg
    T = temperature - 273.15 # K -> C
    W = H2O_pressure * 7.50061683 # kPa -> mmHg
    sigma_squared = 1.0 / ((λ*1e-9) * 1e6)^2.0 # inverse wavenumber squared in micron^-2
    n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))+ (255.4 / (41.0 - sigma_squared))) * 1e-6
    n_minus_one *= P * (1.0 + (1.049 - 0.0157 * T) * 1e-6 * P) / (720.883 * (1.0 + 0.003661 * T))
    n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * T) * W * 1e-6
    return n_minus_one
end

function get_refraction(λ, ζ; pressure=69.328, temperature=293.15, H2O_pressure=1.067)
    """Compute the angle of refraction for a photon entering the atmosphere.

    Photons refract when transitioning from space, where the refractive index n = 1.0 exactly, to
    air, where the refractive index is slightly greater than 1.0.  This function computes the
    change in zenith angle for a photon with a given wavelength.  Output is a positive number of
    radians, even though the apparent zenith angle technically decreases due to this effect.

    Parameters:
        wave:            Wavelength array in nm
        zenith_angle:    `Angle` from object to zenith (degrees)
        **kwargs:        Keyword arguments to pass to air_refractive_index() to override default
                         pressure, temperature, and/or H2O_pressure.

    Returns:
        the absolute value of change in zenith angle in radians.
    """
    nm1 = air_refractive_index_minus_one(λ, pressure = pressure, temperature = temperature, H2O_pressure=H2O_pressure)
    # The following line is equivalent to:
    # n_squared = (nm1 + 1)**2
    # r0 = (n_squared - 1.0) / (2.0 * n_squared)
    r0 = nm1 * (nm1+2) / 2.0 / (nm1^2 + 2*nm1 + 1)
    return r0 * tan(ζ*pi/180)
end

function refraction_at_layer_pix(atmosphere, observations)
    FTYPE = gettype(atmosphere)
    Δpix = zeros(FTYPE, atmosphere.nlayers, atmosphere.nλ)
    θref = get_refraction(atmosphere.λ_ref, observations.ζ)
    for w=1:atmosphere.nλ
        θλ = get_refraction(atmosphere.λ[w], observations.ζ)
        for l=1:atmosphere.nlayers
            Δpix[l, w] = ((θλ - θref) * (atmosphere.heights[l] * 1000)) / atmosphere.sampling_nyquist_mperpix[l]
        end
    end

    return Δpix
end

function refraction_at_detector_pix(atmosphere, observations; build_dim=observations.dim)
    FTYPE = gettype(atmosphere)
    Δpix = zeros(FTYPE, atmosphere.nλ)
    θref = get_refraction(atmosphere.λ_ref, observations.ζ)
    for w=1:atmosphere.nλ
        θλ = get_refraction(atmosphere.λ[w], observations.ζ)
        Δpix[w] = FTYPE((θλ - θref) * 206265 / observations.detector.pixscale)
    end

    return Δpix
end

@views function propagate_layers(Uin, λ, delta1, deltan, heights, t; FTYPE=Float64)
    N = size(Uin, 1)
    nx = (-N÷2:N÷2-1)' .* ones(FTYPE, N)
    ny = nx'
    k = 2pi / λ
    nlayers = length(heights)
    Δh = (heights[2:end] .- heights[1:nlayers-1]) .* 1000
    α = heights ./ heights[nlayers]
    δ = (1 .- α) .* delta1 .+ α .* deltan
    m = δ[2:nlayers] ./ δ[1:nlayers-1]

    for l=1:nlayers-1
        if Δh[l] == 0
            continue
        else
            δf = 1 / (N .* δ[l])
            fX = nx .* δf
            fY = ny .* δf
            fsq = fX.^2 .+ fY.^2
            δz = Δh[l]
            Q2 = cis.(-(2pi^2 * δz / m[l] / k) .* fsq)

            δx = δ[l]
            Ufreq = ft(Uin ./ m[l]) .* δx^2
            Uprop = Q2 .* Ufreq
            Ux = ift(Uprop) .* (N*δf)^2
            Uin .= t[:, :, l+1] .* Ux
        end
    end

    if Δh[nlayers-1] == 0
        Uout = Uin
    else
        xn = nx .* δ[nlayers]
        yn = ny .* δ[nlayers]
        rnsq = xn.^2 .+ yn.^2
        Q3 = cis.((k/2 * (m[nlayers-1] - 1) / (m[nlayers-1]*Δh[nlayers-1])) .* rnsq)
        Uout = Q3 .* Uin
    end

    return Uout
end

function CN2_huffnagel_valley_generalized(h; A=1.7e-14, hA=100, B=2.7e-16, hB=1500, C=3.59e-53, hC=1000, D=0, d=1, hD=0)
    # Everything in meters - default is HV 5-7
    # A is the coefficient for the surface (boundary layer) turbulence strength (m−2/3 )
    # hA is the height for its 1/e decay (meters)
    # B and hB similarly define the turbulence in the troposphere (up to about 10 km)
    # C and HC define the turbulence peak at the tropopause
    # D and HD define one or more isolated layers of turbulence, with d being the layer thickness (meters).
    return A * exp(-h/hA) +  B * exp(-h/hB) + C * (h^10) * exp(-h/hC) + D * exp(-(h-hD)^2 / (2 * d^2))
end

function wind_profile_greenwood(h, ζ)
    v = 8 .+ 30 .* exp.(-( (h .* cos(ζ*pi/180) .- 9.4) ./ 4.8 ).^2 )
    return v
end

function wind_profile_roberts2011(h, ζ; A₀=5.0, A₁=29.6, A₂=12.335, A₃=3.405)#, site, month)
    # A₀, A₁, A₂, A₃ = get_roberts2011_coeffs(site, month)
    v = A₀ .+ A₁ .* exp.( -( (h .* cos(ζ*pi/180) .- A₂) ./ A₃ ).^2 )
    return v
end

function interpolate_phase(ϕ, λin, λout)
    x = 1:size(ϕ, 1)
    l = 1:size(ϕ, 3)
    itp = interpolate((x, x, l, λin), ϕ, Gridded(Linear()))
    return itp(x, x, l, λout)
end

@views function calculate_smoothed_opd(atmosphere, observations, target_rmse)
    FTYPE = gettype(atmosphere)
    ϕ_ref = (FTYPE(2pi)/atmosphere.λ_ref) .* atmosphere.opd
    ϕ_smooth = zeros(FTYPE, size(ϕ_ref))
    target_rmse_per_layer = target_rmse / sqrt(atmosphere.nlayers)
    for l=1:atmosphere.nlayers
        sampling_ref = atmosphere.sampling_nyquist_mperpix[l] * (atmosphere.λ_ref / atmosphere.λ_nyquist)
        D_ref_pix = round(Int64, observations.D / sampling_ref)
        mask = make_simple_mask(atmosphere.dim, D_ref_pix)
        smooth_to_rmse!(ϕ_smooth[:, :, l], ϕ_ref[:, :, l], target_rmse_per_layer, mask, size(ϕ_ref, 1), FTYPE=FTYPE)
    end

    opd_smooth = ϕ_smooth .* (atmosphere.λ_ref / FTYPE(2pi))
    return opd_smooth
end

function composite_r0_to_layers(r0_target, heights, λ, ζ)
    heights = max.(0.01, heights)
    heights .*= 1000
    Cn2 = CN2_huffnagel_valley_generalized.(heights)
    k = 2pi / (λ .* 1e-9)
    r0(x) = (0.423 * k^2 * sec(ζ*pi/180) * sum(heights .* x))^(-3/5)
    diff = (r0(Cn2) - r0_target) / r0_target
    while abs(diff) > 1e-3
        Cn2 .+= 0.001 .* sign(diff) .* Cn2
        diff = (r0(Cn2) - r0_target) / r0_target
    end
    r0_layers = (0.423 .* k^2 * sec(ζ*pi/180) .* Cn2 .* heights).^(-3/5)
    return r0_layers
end
