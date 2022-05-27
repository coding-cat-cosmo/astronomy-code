# astronomy-code
redshift and DLA detection in python adapted from https://github.com/sbird/gp_dla_detection/tree/zqsos2 \
seven files to run in order are as follows:  
set_parameters.py which creates a parameters.pkl file stored in the directory with all the python scripts.  
download_catalogs.py which creates all the file hierarchy related to the DR9Q DR10Q and DR12Q catalogs for quasar data.  
download_spectra.py which gets all of the raw data in the form of fits files for the quasars.  
preload_qsos.py which reads all of the raw data and puts it into a much nicer form for the Gaussian process to learn on in a file preloaded_qsos. 
learning.py and learning_functions.py which creates the Gaussian process from the filtered quasars and improves it as much as it can and creates a learned_model file afterwards.  
generate_dla_samples.py which makes the Halton sequence and some integrations necessary for the testing and saves it in a dla_samples file.  
testing.py and voigt_function.py which uses all the previously made files together to estimate quasar redshift and DLA detection and redshift and saves it in a processed_qsos file.  

The temporary.py and minimization.py are files used to save everything and just run the minimization if wanted. The temporary.py was used to just get the model creation part done while the minimization.py actually optimizes and saves the model.\
The combination.py combines different jobs together if that is perferred since the testing script supports multiprocessing.\
Lines 739-766 can be changed to allow multiprocessing in the testing.py file with changing the unpacking method used for the variables.
