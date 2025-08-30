import Interpolations: LinearInterpolation, Line


const OPTICS_DIR = "$(@__DIR__)/../data/optics"
const ELEMENT_FILENAMES = Dict(
    ## Filter
    "Bessell:U"=>"$(OPTICS_DIR)/Generic_Bessell.U.dat",
    "Bessell:B"=>"$(OPTICS_DIR)/Generic_Bessell.B.dat",
    "Bessell:V"=>"$(OPTICS_DIR)/Generic_Bessell.V.dat",
    "Bessell:R"=>"$(OPTICS_DIR)/Generic_Bessell.R.dat",
    "Bessell:I"=>"$(OPTICS_DIR)/Generic_Bessell.I.dat",
    ## Lens Coatings
    "Thorlabs:A"=>"$(OPTICS_DIR)/thorlabs-A.dat",
    "Thorlabs:B"=>"$(OPTICS_DIR)/thorlabs-B.dat",
    "Thorlabs:AB"=>"$(OPTICS_DIR)/thorlabs-AB.dat",
    "Thorlabs:MLA-AR"=>"$(OPTICS_DIR)/thorlabs-mla-ar.dat",
    ## Mirror Coatings
    "Thorlabs:OAP-P01"=>"$(OPTICS_DIR)/thorlabs-OAP-45AOI-P01.dat",
    "Thorlabs:Plano-P01"=>"$(OPTICS_DIR)/thorlabs-plano-45AOI-P01.dat",
    ## Dichroics
    "Thorlabs:DMLP650P-transmitted"=>"$(OPTICS_DIR)/DMLP650-transmitted.dat",
    "Thorlabs:DMLP650P-reflected"=>"$(OPTICS_DIR)/DMLP650-reflected.dat",
    "Thorlabs:DMLP805P-transmitted"=>"$(OPTICS_DIR)/DMLP805-transmitted.dat",
    "Thorlabs:DMLP805P-reflected"=>"$(OPTICS_DIR)/DMLP805-reflected.dat",
)


struct OpticalElement{T<:AbstractFloat}
    name::String
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalElement(;
            λ=[],
            response=[],
            xflip=false,
            yflip=false,
            name="",
            FTYPE=Float64
        )
        """
            OpticalElement(; λ=..., response=..., xflip=..., yflip=..., name=..., FTYPE=...)

        Creates an `OpticalElement` structure. If `name` is specified and matches an entry in `ELEMENT_FILENAMES`
        it will read the spectral response of the object from the corresponding file. `λ` and `response` can be 
        specified manually to create an optical element.

        * `xflip` can be specified to cause the element to flip the image about the y axis
        * `yflip` can be specified to cause the element to flip the image about the x axis
        * `FTYPE` can be specified to change the Floating-point precision
        """
        if name != ""
            λ₀, response = readfile(ELEMENT_FILENAMES["$name"])
            λ₀ .*= 1e-9
            if λ != []
                itp = LinearInterpolation(λ₀, response, extrapolation_bc=Line())
                response = max.(0.0, itp(λ))
            else
                λ = λ₀
            end
        end

        return new{FTYPE}(name, λ, response, xflip, yflip)
    end
end

struct Detector{T<:AbstractFloat, S<:Real}
    label::String
    λ::Vector{T}
    λ_nyquist::T
    qe::Vector{T}
    rn::T
    gain::T
    saturation::T
    pixscale::T
    exptime::T
    function Detector(;
            λ=[],
            λ_nyquist=400.0,
            pixscale=0.0,
            qe=[1.0],
            rn=0.0,
            gain=1.0,
            saturation=1e99,
            exptime=5e-3,
            label="",
            verb=true,
            FTYPE=Float64,
            DTYPE=FTYPE
        )
        detector = new{FTYPE, DTYPE}(label, λ, λ_nyquist, qe, rn, gain, saturation, pixscale, exptime)
        if verb == true
            display(detector)
        end
        return detector
    end
end

struct OpticalSystem{T<:AbstractFloat}
    elements::Vector{OpticalElement{T}}
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalSystem(
            elements,
            λ;
            verb=true,
            FTYPE=Float64
        )

        xflip = false
        yflip = false
        response = ones(FTYPE, length(λ))
        for element in elements
            itp = LinearInterpolation(element.λ, element.response, extrapolation_bc=Line())
            response .*= max.(0.0, itp(λ))
            xflip = xflip ⊻ element.xflip
            yflip = yflip ⊻ element.yflip
        end

        optical_system =  new{FTYPE}(elements, λ, response, xflip, yflip)
        if verb == true
            display(optical_system)
        end
        return optical_system
    end
end

mutable struct Observations{T<:AbstractFloat, S<:Real}
    label::String
    masks::Masks{T}
    optics::OpticalSystem{T}
    phase_static::Array{T, 3}
    diversity::Union{Diversity{T}, Nothing}
    detector::Detector{T, S}
    ζ::T
    D::T
    aperture_area::T
    times::Vector{T}
    nsubexp::Int64
    nepochs::Int64
    nsubaps::Int64
    nsubaps_side::Int64
    dim::Int64
    images::Array{T, 4}
    entropy::Matrix{T}
    psfs::Array{T, 3}
    model_images::Array{T, 4}
    w::Vector{Int64}
    positions::Array{T, 4}
    function Observations(
            optics,
            detector;
            ζ=Inf,
            D=Inf,
            D_inner_frac=0,
            area=pi*(D/2)^2*(1-D_inner_frac^2),
            times=[],
            nsubexp=-1,
            nepochs=0,
            nsubaps_side=1,
            dim=0,
            ϕ_static=[;;;],
            diversity=nothing,
            build_dim=dim,
            label="",
            verb=true,
            FTYPE=Float64
        )

        masks = Masks(
            build_dim,
            detector.λ,
            nsubaps_side=nsubaps_side, 
            D_inner_frac=D_inner_frac,  
            λ_nyquist=detector.λ_nyquist, 
            verb=verb,
            label=label,
            FTYPE=FTYPE
        )
        if !isnothing(diversity)
            create_diversity_phase!(diversity, masks)
        end
        nsubaps = masks.nsubaps
        nepochs = (nepochs==0) ? length(times) : nepochs
        DTYPE = gettypes(detector)[2]
        optics.response .*= detector.qe
        observations = new{FTYPE, DTYPE}(label, masks, optics, ϕ_static, diversity, detector, ζ, D, area, times, nsubexp, nepochs, nsubaps, nsubaps_side, dim)
        if verb == true
            display(observations)
        end
        return observations
    end
    function Observations(
            times,
            images,
            optics,
            detector;
            ζ=Inf,
            D=Inf,
            D_inner_frac=0,
            area=pi*(D/2)^2*(1-D_inner_frac^2),
            nsubaps_side=1,
            nsubexp=-1,
            ϕ_static=[;;;],
            diversity=nothing,
            build_dim=size(images, 1),
            label="",
            verb=true,
            FTYPE=Float64
        )
        dim, ~, nsubaps, nepochs = size(images)
        masks = Masks(
            build_dim,
            detector.λ,
            nsubaps_side=nsubaps_side, 
            D_inner_frac=D_inner_frac,  
            λ_nyquist=detector.λ_nyquist, 
            verb=verb,
            label=label,
            FTYPE=FTYPE
        )
        if !isnothing(diversity)
            create_diversity_phase!(diversity, masks)
        end
        entropy = [calculate_entropy(images[:, :, n, t]) for n=1:nsubaps, t=1:nepochs]
        DTYPE = gettypes(detector)[2]
        optics.response .*= detector.qe
        observations = new{FTYPE, DTYPE}(label, masks, optics, ϕ_static, diversity, detector, ζ, D, area, times, nsubexp, nepochs, nsubaps, nsubaps_side, dim, images, entropy)
        if verb == true
            display(observations)
        end
        return observations
    end
end

@views function calculate_wfs_slopes(observations_wfs)
    FTYPE = gettype(observations_wfs)
    nsubaps = observations_wfs.nsubaps
    nepochs = observations_wfs.nepochs
    ∇ϕx = zeros(FTYPE, nsubaps, nepochs)
    ∇ϕy = zeros(FTYPE, nsubaps, nepochs)
    # composite_image = dropdims(sum(observations_wfs.images, dims=(3, 4)), dims=(3, 4))
    for n=1:nsubaps
        for t=1:nepochs
            Δy, Δx = Tuple(argmax(ccorr_psf(composite_image, observations_wfs.images[:, :, n, t])))
            ∇ϕx[n, t] = Δx * observations_wfs.D / observations_wfs.nsubaps_side
            ∇ϕy[n, t] = Δy * observations_wfs.D / observations_wfs.nsubaps_side
        end
    end
    return ∇ϕx, ∇ϕy
end
