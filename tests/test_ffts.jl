include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Test
using FFTW
using InteractiveUtils


@testset "Pre-allocated FFT operations" begin
    @testset "Real-valued input" begin
        dim = 256
        FTYPE = Float32

        ## test - IFFT(FFT(a)) = a; a ∈ ℝ
        ffts, outFFT = setup_fft(FTYPE, dim)
        iffts, outIFFT = setup_ifft(FTYPE, dim)
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        ffts(outFFT, a)
        iffts(outIFFT, outFFT)
        @test outIFFT ./ dim ≈ ifft(fft(a)) ≈ a

        ## test - sum(a ⊙ b) = sum(a); a,b ∈ ℝ, ∑b = 1
        convs = setup_conv(FTYPE, dim)
        a = rand(FTYPE, dim, dim)
        a ./= sum(a)
        b = rand(FTYPE, dim, dim)
        b ./= sum(b)
        outCONV = zeros(FTYPE, dim, dim)
        convs(outCONV, a, b)
        @test sum(outCONV) ≈ sum(a)
        @test outCONV ≈ ifft(fft(a) .* fft(b))
        
        ## test - a ⊙ preconv(b) = a ⊙ b; a,b ∈ ℝ, ∑b = 1
        fullCONV = deepcopy(outCONV)
        preconv = preconvolve(b)
        preconv(outCONV, a)
        @test outCONV ≈ fullCONV ≈ ifft(fft(a) .* fft(b))

        ## test - sum(a ⊗ b) = sum(a); a,b ∈ ℝ, ∑b = 1
        corrs = setup_corr(FTYPE, dim)
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        b = rand(FTYPE, dim, dim)
        b ./= sum(b)
        outCORR = zeros(FTYPE, dim, dim)
        corrs(outCORR, a, b)
        @test sum(outCORR) ≈ sum(a)
        @test outCORR ≈ ifft(fft(a) .* conj.(fft(b)))
        
        ## test - a ⊗ preconv(b) = a ⊗ b; a,b ∈ ℝ, ∑b = 1
        fullCORR = deepcopy(outCORR)
        precorr = precorrelate(b)
        precorr(outCORR, a)
        @test outCORR ≈ fullCORR ≈ ifft(fft(a) .* conj.(fft(b)))

        ## test - FT(|IFT(a)|²) = a ⊗ a; a ∈ ℝ
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        a ./= sum(a)
        outAUTO1 = zeros(FTYPE, dim, dim)
        outAUTO2 = zeros(FTYPE, dim, dim)
        corrs(outAUTO1, a, a)
        autocorr = setup_autocorr(FTYPE, dim)
        autocorr(outAUTO2, a)
        @test outAUTO2 ≈ outAUTO1 ≈ ifft(fft(a) .* conj.(fft(a))) ≈ ifft(abs2.(fft(a)))
    end

    @testset "Complex-valued input" begin
        dim = 256
        FTYPE = Complex{Float32}

        ## test - IFFT(FFT(a)) = a; a ∈ ℝ
        ffts, outFFT = setup_fft(FTYPE, dim)
        iffts, outIFFT = setup_ifft(FTYPE, dim)
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        ffts(outFFT, a)
        iffts(outIFFT, outFFT)
        @test outIFFT ./ dim ≈ ifft(fft(a)) ≈ a

        ## test - sum(a ⊙ b) = sum(a); a,b ∈ ℝ, ∑b = 1
        convs = setup_conv(FTYPE, dim)
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        b = rand(FTYPE, dim, dim)
        b ./= sum(b)
        outCONV = zeros(FTYPE, dim, dim)
        convs(outCONV, a, b)
        @test sum(outCONV) ≈ sum(a)
        @test outCONV ≈ ifft(fft(a) .* fft(b))
        
        ## test - a ⊙ preconv(b) = a ⊙ b; a,b ∈ ℝ, ∑b = 1
        fullCONV = deepcopy(outCONV)
        preconv = preconvolve(b)
        preconv(outCONV, a)
        @test outCONV ≈ fullCONV ≈ ifft(fft(a) .* fft(b))

        ## test - sum(a ⊗ b) = sum(a); a,b ∈ ℝ, ∑b = 1
        corrs = setup_corr(FTYPE, dim)
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        b = rand(FTYPE, dim, dim)
        b ./= sum(b)
        outCORR = zeros(FTYPE, dim, dim)
        corrs(outCORR, a, b)
        @test sum(outCORR) ≈ sum(a)
        @test outCORR ≈ ifft(fft(a) .* conj.(fft(b)))
        
        ## test - a ⊗ preconv(b) = a ⊗ b; a,b ∈ ℝ, ∑b = 1
        fullCORR = deepcopy(outCORR)
        precorr = precorrelate(b)
        precorr(outCORR, a)
        @test outCORR ≈ fullCORR ≈ ifft(fft(a) .* conj.(fft(b)))

        ## test - FT(|IFT(a)|²) = a ⊗ a; a ∈ ℝ
        a = rand(FTYPE, dim, dim) .* rand(1:10000)
        outAUTO1 = zeros(FTYPE, dim, dim)
        outAUTO2 = zeros(FTYPE, dim, dim)
        corrs(outAUTO1, a, a)
        autocorr = setup_autocorr(FTYPE, dim)
        autocorr(outAUTO2, a)
        @test outAUTO2 ≈ outAUTO1 ≈ ifft(fft(a) .* conj.(fft(a))) ≈ ifft(abs2.(fft(a)))
    end
end
