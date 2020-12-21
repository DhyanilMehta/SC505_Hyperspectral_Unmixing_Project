# Hyperspectral Unmixing 

Some of files are present in Git LFS
## How to Clone?
git-lfs url

##What is Hyperspectral Unmixing?
The  images  captured  by  the  HSI  camera,  will  give  us  the  image  data  in  the  form  of  a  hyper-cube.  In the hypercube, the pixels may not be a single pixel, it can be a combination of multiple spectra.  Such pixels are known as mixed-pixels or hyper-pixels.  We need to decompose these hyper-pixels into endmember signatures(pure pixel signatures) and corresponding abundances. The process of decomposition of hyper-pixels is known as Hyperspectral Unmixing. We have here used the Multiplicative Update NMF method for Unmixing.

##Algorithm
procedureHSU(X)
Initialize W and H
Define Convergence Tolerance
StopIter←maxIterations
counter←0
while !Converged do
	Wold←W
	W←W·(XHT)(WHHT)
	H←H·(WToldX)(WToldWoldH)
	counter←counter+1
	Check Convergence
	if counter==StopIter then
		break;
	end if
end while
return W,H
end procedure
