# Hyperspectral Unmixing 

Some of files are present in Git LFS
## How to Clone?
git-lfs url

## What is Hyperspectral Unmixing?
The  images  captured  by  the  HSI  camera,  will  give  us  the  image  data  in  the  form  of  a  hyper-cube.  In the hypercube, the pixels may not be a single pixel, it can be a combination of multiple spectra.  Such pixels are known as mixed-pixels or hyper-pixels.  We need to decompose these hyper-pixels into endmember signatures(pure pixel signatures) and corresponding abundances. The process of decomposition of hyper-pixels is known as Hyperspectral Unmixing. We have here used the Multiplicative Update NMF method for Unmixing.

## Algorithm for Hyperspectral Unmixing using Non-negative Matrix Factorization using Multiplicative Update method

	procedureHSU(A):
		Initialize W and H
		Define Convergence Tolerance
		StopIter ← maxIterations
		counter ← 0
		while !Converged do
			W_old ← W
			H_old ← H
			W ← W · (A x H_old.T)/(W_old x H_old x H_old.T)
			H ← H · (W_old.T x A)/(W_old.T x W_old x H)
			counter ← counter + 1
			Check Convergence
			if counter == StopIter then
				break
			end if
		end while
		return W,H
	end procedure


## How to run the code
This code will run for *Indian Pines*, *SalinasA*, and *Pavia University* hyperspectral datasets. Download the corrected version for Indian Pines and SalinasA. You can download them from this link: [Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). 
After downloading them put it into the dataset folder.

The main file from where you can run then run the project is *NMF_run.py*. Depending on the dataset on which you want to run, uncomment the 3 lines of code in which the name of the desired dataset is written and keep the other two commented and then run the file. By default we have uncommented the Indian pines code.

As an output you will get two graphs and some console output. Console output will show the final unmixed abundance and spectral endmember matrices and a list of error and time taken per outer iteration. One grpah will show the error trend of NMF for the dataset and another will show the plot of unmixed spectral endmembers according to their calculated value for each spectral bands. (NMF will not give a 100% accurate graph of unmixed spectral endmembers)

*NMF_tools.py* contains the helper functions which are used in *NMF_run.py*. *data* folder will contain the list of abundance matrices, endmembers matrices, errors, and time taken all for each outer iteration so that you don't have to run the whole code to get the matrices again.
