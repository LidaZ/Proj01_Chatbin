macro "FFT1D Tool - C099A8877099L0088" {
    profile = getProfile();
    fft=Array.fourier(profile,"None");
    s=newArray(lengthOf(fft));
    for (i=1;i<lengthOf(fft);i++) {
        s[i]=i; // wavenumber = 2*PI/i
    }
    s[0]=0
    Plot.create("Fourier spectrum of the line profile","wavenumber","FFT(line)",s,fft);
    Plot.setLimits(0, 50, 0, 40);
    Plot.show();
}