"""
	Module containing useful functions for analysis of synthetic radio observations.
	The whole script is meant to work under the CGS unit system.
"""

import os, sys, time 
from datetime import datetime
import numpy as np
from astropy.io import fits, ascii
from astropy import units as u
from astropy.constants import c, k_B, h, m_p, m_e, m_n

### Physical constants
c = c.cgs.value
h = h.cgs.value
k_B = k_B.cgs.value
m_H = m_p.cgs.value + m_e.cgs.value
m_D = m_H + m_n.cgs.value
m_H2D = 2 * m_H + m_D - m_e.cgs.value


###	Useful objects
class color:
    header = '\033[95m'
    okblue = '\033[94m'
    okgreen = '\033[92m'
    warning = '\033[93m'
    fail = '\033[91m'
    none = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'

class Lmu10M2:
	def __init__(self):
		self.name = type(self).__name__
		self.runs = ['run_0000','run_0011','run_0023','run_0038','run_0053','run_0068',\
					'run_0083','run_0099','run_0114','run_0129','run_0145']
		self.time = [0.0,15.8,29.8,44.0,59.1,74.1,89.0,104.0,119.0,133.9,148.9]#kyr
		self.sink = 77#kyr


class oH2Dp():
	"""
		Molecular values taken from the Leiden Atomic and Molecular Database  for the transition o-H2D+ 1(10)-1(11).
		Observable values assumed to match an APEX obs.
	"""	 
	def __init__(self):
		self.name = type(self).__name__.lower()
		self.molID = '0001'
		self.nu = 372.4313847e9 #Hz
		self.fwhm = 16.8 #["] APEX
		self.Tex = 16.4 #[K] Tex=Tg
		self.mu = 0.6e-18 # Dipole momment
		self.J = 1
		self.E_J1 = 0  #[K] Energy of the transition from J=1 to J=0
		self.E = 17.87 #[K] Energy of the transition
		self.g = 9	#Statistical weight	# READ PARISE 
		self.A = 1.082e-4 #[s^-1] Einstein coef.
		self.Q = 10.3375
		self.T_CDMS = [2.725,5.000,9.375,18.75,37.50,75.00,150.0,225.0,300.0,500.0]
		self.Q_CDMS = [1.000,1.0000,1.0037,1.2151,3.1037,9.7322,27.5467,49.9611,76.4512,163.9689]

		
###	Useful functions

def print_(string, fname='', verbose=False, bold=False):
	if verbose:
		if bold:
			print(f"{color.bold} [{fname}] {string} {color.none}")
		else:		
			print(f"[{fname}] {string}")


def write_fits(filename, data, header, overwrite, fname='', verbose=False):
	if filename != '':
		if overwrite and os.path.exists(filename):
			print_('overwriting file ...', fname, verbose=verbose)
			os.remove(filename)

		fits.HDUList(fits.PrimaryHDU(data=data, header=header)).writeto(filename) 	

		print_(f"written file {filename}", fname, verbose)


def elapsed_time(runtime, fname='', verbose=False):
	run_time = time.strftime("%H:%M:%S", time.gmtime(runtime))

	print_(f"Elapsed time: {run_time}", fname, verbose)
		

def set_keywords(infile, outfile, overwrite=True, verbose=True, delete=False, **kwargs):
	""" Function to add or delete keywords to or from the header of a fits file. """

	fname = sys._getframe().f_code.co_name
	
	# Check if the file is a fits file or not.
	if not infile.endswith('.fits'):
		raise Exception(f"[{fname}] Input file is not a fitsfile.")

	# Load data and header.
	data, header = fits.getdata(infile, header=True)

	for key, value in kwargs.items():
		if delete:
			# Unpack arguments and delete them from the header.
			print_(f"Deleting keyword {key}", fname, verbose=verbose)
			del header[key]
		else:
			# Unpack keyword arguments and add each to the header.
			print_(f"Adding keyword {key} = {value}", fname, verbose=verbose)
			header[key] = value

	# Write data to fits
	write_fits(outfile, data, header, overwrite, fname, verbose)	


def trim_cube(infile, outfile='alma+aca_3_sigma.fits', sigma=3, overwrite=True, verbose=False):
	"""
		Just a temporal function to trim data cubes with a lower threshold
	"""
	fname = sys._getframe().f_code.co_name

	cube = fits.getdata(infile)
	header = fits.getheader(infile)

	rms = get_rms(cube, verbose=True) 
	cube[cube < sigma*rms] = float('NaN') 

	write_fits(outfile, cube, header, overwrite, fname, verbose)	


def get_rms(infile, verbose=False):
	"""
		Calculate the standard deviation of an fits file
		by obtaining it within four rectangular regions
		corresponding to the borders of the image, and 
		then averaging them all.

		It assumes data comes in Jy/beam by now.
		It assumed data comes from a fits file.
	"""

	fname = sys._getframe().f_code.co_name
	start_time = time.time()
	
	# Detects whether input data comes from a fits file or an array
	if isinstance(infile, str):

		input_from_file = True
		data = fits.getdata(infile)
		header = fits.getheader(infile)
		size_x = header['NAXIS1']
		size_y = header['NAXIS2']

	elif isinstance(infile, (list, np.ndarray)):

		input_from_file = False
		data = np.array(infile)
		shape = np.shape(data) 
		size_x = shape[1]
		size_y = shape[2]

	# To do: merge the following if into a function to reduce code lines.
	regions = np.zeros(4)
	if np.ndim(data) == 3:

		# Left upright rectangle
		regions[0] = np.nanstd(data[:, int(size_x*0.0) : int(size_x*0.1), int(size_y*0.0) : int(size_y*1.0)]) 

		# Bottom horizontal rectangle
		regions[1] = np.nanstd(data[:, int(size_x*0.0) : int(size_x*1.0), int(size_y*0.0) : int(size_y*0.1)]) 

		# Right upright rectangle
		regions[2] = np.nanstd(data[:, int(size_x*0.9) : int(size_x*1.0), int(size_y*0.0) : int(size_y*1.0)]) 

		# Top horizontal rectangle
		regions[3] = np.nanstd(data[:, int(size_x*0.0) : int(size_x*1.0), int(size_y*0.9) : int(size_y*1.0)]) 

	elif np.ndim(data) == 2:

		# Left upright rectangle
		regions[0] = np.nanstd(data[int(size_x*0.0) : int(size_x*0.1), int(size_y*0.0) : int(size_y*1.0)]) 

		# Bottom horizontal rectangle
		regions[1] = np.nanstd(data[int(size_x*0.0) : int(size_x*1.0), int(size_y*0.0) : int(size_y*0.1)]) 

		# Right upright rectangle
		regions[2] = np.nanstd(data[int(size_x*0.9) : int(size_x*1.0), int(size_y*0.0) : int(size_y*1.0)]) 

		# Top horizontal rectangle
		regions[3] = np.nanstd(data[int(size_x*0.0) : int(size_x*1.0), int(size_y*0.9) : int(size_y*1.0)]) 

	else:
		print_("Wrong number of dimensions in data. Must be either 2 or 3D.", fname, verbose=verbose, bold=True)
		return 0

	# Average over all four regions
	rms = np.nanmean(regions)

	print_(f'cube rms: {rms} Jy/beam', fname=fname, verbose=verbose)

	return rms


def get_N_mod(runfile, d=(1,'kpc'), bmaj=(1.132,'arcsec'), width=None, center='max', cmap='magma', resolution=512, fontsize=13, title=None, vmin=None, vmax=None, savefig='', show=True, mask=True, verbose=False):
	"""
	Obtain the column density of oH2Dp from the model file via averaging 
	over the volume defined by region and radius.

	[Parameters]
	runfile	: (string)	Name of the hdf5 file that contains the model snapshot.
	d 		: (tuple)	Artificial distance to the source used to convert angular scales into a physical width. Must be a tuple (val, "unit").
	bmaj	: (tuple)	Value of an angular scale to be converted into a map width. Must be a tuple (val, "unit").
	width	: (tuple)	Width of half of the map in physical scales. If width is provided, it overrides bmaj and d parameters. Must be a tuple (val, "unit").
	cmap 	: (string)	Colorscheme to be used in the colorbar.
	resolution	: (int)	Number of pixels per side in the resulting map.
	fontsize: (float)	Size of the plot labels and title.
	title	: (string)	Optional title for the plot.
	vmin	: (float)	Minimun value for the plot colorbar.
	vmin	: (float)	Maximum value for the plot colorbar.
	savefig	: (string)	Optional filename to save the plot figure.
	show	: (boolean)	Whether to show the plot or not.
	mask	: (boolean)	Whether to apply a circular mask or not.
	"""

	import yt

	# Suppress INFO messages setting the log level to "ERROR"
	yt.funcs.mylog.setLevel(50) 
	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	def circle_mask(h,w, center=None, radius=None):
		"""
			applies a circular mask to the input data 
			aiming to mimic the telescope beam area.
		"""
		if center is None:
			center = [int(w/2), int(h/2)]
		if radius is None:
			radius = min(center[0], center[1], w-center[0], h-center[1])

		Y, X = np.ogrid[:h, :w]
		r = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

		mask = r > radius

		return mask

	# Create a new field containing the oH2D+ number density
	def n_oh2dp(field, data):
		import yt.units as u
		n = data['dens'] * data['h2do'] / (m_H2D*u.g)
		return n

	def n_tot(field, data):
		import yt.units as u
		return data['dens'] / (2.4 * m_H*u.g)

	             
	# Validate bmaj parameter type                                                                   
	if not isinstance(bmaj, (tuple)):
		raise ValueError('Not a valid type for bmaj. Must be a tuple (val, "unit").')
	else:
		if str(bmaj[1]).lower() == 'arcsec':
			bmaj = bmaj[0] *u.arcsec.to(u.rad) 

		elif str(bmaj[1]).lower() == 'deg':
			bmaj = bmaj[0] *u.deg.to(u.rad)               

		elif str(bmaj[1]).lower() == 'rad':
			bmaj = bmaj[0]                

	# Validate d parameter type
	if not isinstance(d, (tuple)):
		raise ValueError('Not a valid type for d. Must be a tuple (val, "unit").')
	else:
		if str(d[1]).lower() == 'kpc':
			d = d[0] *u.kpc.to(u.cm) 

		elif str(d[1]).lower() == 'pc':
			d = d[0] *u.pc.to(u.cm) 

		elif str(d[1]).lower() == 'au':
			d = d[0] *u.au.to(u.cm)               

		elif str(d[1]).lower() == 'm':
			d = d[0]*u.m.to(u.cm)                

		elif str(d[1]).lower() == 'cm':
			d = d[0]      

	# Validate width parameter type
	if width == None:
		# In case width is not provided, calculate it based on bmaj and d
		width = (bmaj*d, 'cm')                     

	elif isinstance(width, (tuple)) and \
		type(width[0]) in (float, int) and \
		type(width[1]) == str:

		width = width

	else:
		raise ValueError('Not a valid type for width. Must be a tuple (val, "unit").')	


	# Load the data from the snapshot file
	ds = yt.load(runfile)                                                  

	d = ds.all_data()
	d_trimmed = d.cut_region(["(obj['density'] > 4.008e-20) & (obj['density'] < 4.008e-18)"])
	yt.add_field(('gas','n_oh2dp'), function=n_oh2dp, units='cm**-3', force_override=True)
	yt.add_field(('gas','n_tot'), function=n_tot, units='cm**-3', force_override=True)

	# Projection of data                       
	axis = 'z'
	field = 'n_oh2dp'
	weight = None
	data_trimmed = None
	#data_trimmed = d_trimmed
	
	# info about method and weight_field parameters
	# at https://yt-project.org/doc/visualizing/plots.html
	proj = ds.proj(field, axis, weight_field=weight, data_source=data_trimmed, method='integrate')

	if center in ['max', 'm']:
		# center on the density peak
		center = ds.find_max(field)[1]
	elif center in ['center','c']:
		# center on the box center
		center = [0,0,0]



	# Resolution of the numpy grid
	resolution = np.array([resolution,resolution])

	# Convert grid data into a 2D grid for easy plotting
	column = proj.to_frb(width, resolution, center)                    
	                                                                    
	# Compute the column density                                                                    
	cdensity = np.array(column[field])                           


	# Evolutionary time of the snapshot
	time = np.round(ds.current_time.in_units('kyr'), 1)

	cdensity = np.flipud(cdensity)	

	if mask:
		# create a circular mask to mimic the telescope beam area
		Mask = circle_mask(cdensity.shape[0], cdensity.shape[1])
		cdensity[Mask] = 0

	# Print the column density averaged over the map area
	N_mod = np.nanmean(cdensity)    
	print_(f"N_mod [max, mean]:\t{cdensity.max()}\t{N_mod}", fname, verbose=verbose)

	if show:
		import matplotlib.pyplot as plt
		# Plotting commands
		plt.imshow(np.log10(cdensity), cmap=cmap, vmin=vmin, vmax=vmax)                          
		plt.colorbar().set_label(r'$N\,$[o-H$_2$D$^+$] / cm$^{-2}$', size=fontsize)
		#plt.colorbar().set_label(r'$N_{\rm total}$ / cm$^{-2}$', size=fontsize)

		nticks = 3
		width = (np.round(width[0]*u.cm.to(u.au), 1), 'AU')
		plt.xticks(np.linspace(0,resolution.max(),nticks), np.linspace(-width[0]/2.,width[0]/2.,nticks))
		plt.yticks(np.linspace(0,resolution.max(),nticks), np.linspace(-width[0]/2.,width[0]/2.,nticks))              
		plt.xlabel(r'x ['+width[1].upper()+']', size=fontsize)        
		plt.ylabel(r'y ['+width[1].upper()+']', size=fontsize)        

		if title in ['', None]: 
			plt.title(f't = {time} kyr', size=fontsize)
		else:
			plt.title(title, size=fontsize)

		if savefig != '':       
		    plt.savefig(savefig)
		              
		plt.show(block=True)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)	
	            
	return cdensity


def brightness_temperature(infile, outfile='', unit='jy/beam', nu=0, fwhm=0, pixel_size=0, overwrite=False, verbose=False):
	"""
	Convert intensities [Jy/beam] or fluxes [Jy] into brightness temperatures [K].
	Frequencies must be in Hz and fwhm in arcseconds.
	"""
	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	# Detects whether input data comes from a file or an array
	if type(infile) == str:
		input_from_file = True
		header = fits.getheader(infile)
		data = fits.getdata(infile)
		
	elif isinstance(infile, (int, float, list, tuple, np.ndarray)):
		input_from_file = False
		data = np.array(infile) 
		header = {}
		# Ensure data has at least one dimension
		data = np.array(data, ndmin=1)		

		if fwhm != 0:
			if type(fwhm) in [int, float, list, tuple, np.ndarray]:
				print_(f"FWHM: {fwhm}", fname, verbose=verbose)
		else:
			fwhm = np.float64(input('No FWHM found in the header. Enter FWHM["]: '))
		
	else:
		raise IOError(f"[{fname}] Wrong input data format.")


	# Assess the type of fwhm
	if input_from_file:
		if fwhm != 0:
			if type(fwhm) in [int, float, list, tuple, np.ndarray] :
				pass

		elif 'BMAJ' and 'BMIN' in header:
			print_(f"Reading BMIN and BMAJ from header: {fwhm}", fname, verbose=verbose)
			fwhm = [header['BMIN']*u.deg.to(u.arcsec) , header['BMAJ']*u.deg.to(u.arcsec)] 

		elif 'FWHM' in header:
			print_(f"Reading FWHM from header: {fwhm}", fname, verbose=verbose)
			fwhm = np.float32(header['FWHM'])

		else:
			fwhm = np.float64(input('No FWHM found in the header. Enter FWHM["]: '))

	# Read data unit from the header if possible
	unit = header['BUNIT'] if 'BUNIT' in header else unit

	# Convert fwhm from arcseconds to radians. If a scalar, use it both as bmin and bmaj.
	fwhm = np.array(fwhm) * u.arcsec.to(u.rad)
	fwhm = np.prod(fwhm) if np.size(fwhm) > 1 else fwhm**2

	# data = data * (u.erg*u.s**-1*u.cm**-2*u.Hz**-1).to(u.Jy)
	data = data * u.Jy.to(u.erg*u.s**-1*u.cm**-2*u.Hz**-1)

	T_b=[]

	for chan, flux in enumerate(data):

		if 'jy/beam' in unit.lower():

			# Area of a Gaussian beam
			beam_area = ((np.pi * fwhm) / (4*np.log(2)))
			T = (c**2 / (2*k_B*nu**2)) * (flux / beam_area)

		elif 'jy/pix' in unit.lower():

			# Area of a squared beam
			beam_area = (pixel_size*u.arcsec.to(u.rad))**2
			T = (c**2 / (2*k_B*nu**2)) * (flux / beam_area)

		elif unit.lower() == 'jy':

			# Area of a Gaussian beam
			beam_area = ((np.pi * fwhm) / (4*np.log(2)))
			T = (c**2 / (2*k_B*nu**2)) * (flux / beam_area)

		else:	
			raise ValueError(f"{unit} is not a valid unit. Must be given in Jy/beam, Jy/pixel or Jy.")

		T_b.append(T)


	# Write data to fits file if required
	write_fits(outfile, T_b, header, overwrite, fname, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return np.array(T_b)
	

def column_density(infile, outfile='', mol='', nu=0, fwhm=0, Tex=15, mu=0, J=1, E=0, g=0, A=0, Q=0, d_v=None, chans=None, threshold=None, fliplr=False, flipud=False, overwrite=False, verbose=False):
	"""
	Reads intensity or flux density [Jy] values. 
	Intensity must be in [Jy/beam], flux density must be in [Jy].
	Optical depth can be provided by a fits file passing the od_file argument.
	If none is given, will be calculated from the brightness temperature.
	Function is based on cgs unit system.
	"""
	fname = sys._getframe().f_code.co_name
	start_time = time.time()
	
	# Detects whether input data comes from a file or an array
	if type(infile) == str:
		input_from_file = True
		data = fits.getdata(infile)
		header = fits.getheader(infile)

	elif isinstance(infile, (list, np.ndarray)):
		input_from_file = False
		data = np.array(infile) 
		header = {}
		# Ensure data has at least one dimension
		data = np.array(data, ndmin=1)		

	# so the loop iterates over it
	elif isinstance(infile, (int, float)):
		input_from_file = False
		if np.shape(data) == ():
			data = np.array([data])		
			header = {}

	
	# Check the origin of the FWHM value
	if input_from_file:
		#  1) Check if provided as an argument
		if fwhm != 0:
			theta = fwhm
			print_(f"Using BMIN, BMAJ: {theta} arcsec", fname, verbose=verbose)

		# 2) Check if provided in the header
		elif 'BMAJ' in header and 'BMIN' in header:
			theta = [np.float32(header['BMIN'])*u.deg.to(u.arcsec), np.float32(header['BMAJ'])*u.deg.to(u.arcsec)]
			print_(f"Reading BMIN,BMAJ from header: {theta} arcsec", fname, verbose=verbose)

		elif 'FWHM' in header:
			theta = np.float32(header['FWHM'])
			print_(f"Reading FWHM from header: {theta} arcsec", fname, verbose=verbose)

		# 3) Else, ask the user to enter it	
		else:
			theta = np.float32(input('No FWHM found in the header. Enter FWHM["]: '))

	else:
		if fwhm == 0:
			theta = np.float32(input('No FWHM found in the header. Enter FWHM["]: '))


	# Read the rest frequency from the header
	nu = np.float32(header['FREQ']) if ('FREQ' in header and nu != 0) else nu


	# Check the units of the velocity width
	if input_from_file:
		if header['CUNIT3'] == 'km/s':
			d_v = np.abs(header['CDELT3']*(u.km/u.s).to(u.cm/u.s))
		
		elif header['CUNIT3'] == 'm/s':
			d_v = np.abs(header['CDELT3']*(u.m/u.s).to(u.cm/u.s))

		elif header['CUNIT3'] == 'cm/s':
			d_v = np.abs(header['CDELT3'])

	elif not input_from_file and d_v == None:
		d_v = np.abs(float(input("Enter the spectral resolution [km/s]: "))) * (u.km/u.s).to(u.cm/u.s)

	else:
		print_(f"using d_v = {d_v} cm/s.", fname, verbose=verbose)		
		

	# Mask values lower than threshold if provided
	threshold = threshold * get_rms(data) if isinstance(threshold, int) else threshold
	data = np.where(data > threshold, data, float('NaN')) if threshold != None else data	

	# Check whether infile comes in intensity or temperature
	# This function assumed data in temperature units unless specified in the header.
	if input_from_file:

		if header['BTYPE'].lower() == 'intensity':
			if 'jy/beam' in header['BUNIT'].lower():
				data = brightness_temperature(data, unit='jy/beam', nu=nu, fwhm=theta)

			elif 'jy/pixel' in header['BUNIT'].lower():
				pix_size = header['CDELT1']*u.deg.to(u.arcsec)

				print_(f'using pixel_size: {pix_size}', fname, verbose=verbose)

				data = brightness_temperature(data, unit='jy/pixel', nu=nu, fwhm=theta, pixel_size=pix_size)

		else: 
			print_("Assuming input data comes in K.", fname, verbose=verbose)		


	# Slice the cube to integrate only over the required channels
	print_channels_used = lambda c1,c2: print_(f"integrating over channels {c1} to {c2}", fname, verbose=verbose)

	if chans is not None and isinstance(chans,str):

			if ':' in chans:
				chan = chans.split(':')
				print_channels_used (chan[0], chan[1])

			elif '~' in chans:
				chan = chans.split('~')
				print_channels_used (chan[0], chan[1])

			elif '-' in chans:
				chan = chans.split('-')
				print_channels_used (chan[0], chan[1])

			elif ',' in chans:
				chan = chans.split(',')
				print_channels_used (chan[0], chan[1])

			else:
				print_("channel range must separated by either a colon, a dash or a comma. \
				Using the entire spectrum ...", fname, verbose=verbose)

			# Slice the cube
			data = data[int(chan[0]) : int(chan[1])]


	int_tau = []
	Tbg = 2.7 # K (CMB temp)

	# Radiation Temperature
	Jv = lambda nu,T: (h*nu/k_B) / (np.exp((h*nu)/(k_B*T))-1)

	# Loop over the cube 
	for (chan, Tmb) in enumerate(data):

		# Calculate the optical depth
		tau = -np.log(1 - (Tmb / (Jv(nu,Tex)-Jv(nu,Tbg))))

		# Weight by the velocity bin
		tau_i = tau * d_v

		int_tau.append(tau_i)

	# Integrate op. depth over velocity
	int_tau = np.nansum(int_tau, axis=0)

	# Convert the zeros created by np.nansum() back to NaN
	int_tau = np.where(int_tau != 0, int_tau, float('NaN')) if threshold != None else int_tau	

	# Calculate the total column density. (Vastel et al. (2006); Magnum & Shirley (2008))
	col_density = ((8.*np.pi*nu**3)/c**3) * (Q/(g*A)) * (np.exp(E/Tex)/(np.exp((h*nu)/(k_B*Tex))-1)) * int_tau

	# Add brightness type FITS keyword
	if input_from_file: header['BTYPE'] = 'column density'
	
	# Add brightness unit FITS keyword
	if input_from_file: header['BUNIT'] = 'cm**-2'

	# Flip left-right if indicated
	col_density = np.fliplr(col_density) if fliplr else col_density

	# Flip upside down if indicated
	col_density = np.flipud(col_density) if flipud else col_density

	# Write data to fits file if required
	write_fits(outfile, col_density, header, overwrite, fname, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return col_density


def apply_beam(infile, fwhm, pa=0, nu=0, outfile='', bin_img=1, bin_spec=1, eff=1, telescope='APEX', pixel_size=None, output_Tmb=False, flip_lr=False, flip_ud=False, log_scale=False, overwrite=False, verbose=False):
	"""
	Function used to convolve images by single-beams.
	This does not apply for interferometers, in such case, use simobserve instead.

	Bin the image if required (i.e. bin_size!=0).
	Perform convolution with a gaussian beam
	Note: is better to implement convolve_fft() instead of 
	convolve() since it works faster for large kernels (n>500)
	Also rescale intensities from Jy/px to Jy/beam by the (1.331*(fwhm**2)/(pixel_size**2)) factor
	"""
	from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
	from astropy.nddata.utils import block_reduce

	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	# Detects whether input data comes from a file or an array
	if type(infile) == str:
		header = fits.getheader(infile)
		data = fits.getdata(infile)
		input_from_file = True

	elif isinstance(infile, (list, np.ndarray)):
		input_from_file = False
		data = np.array(infile) 
		# Ensure data has at least one dimension
		data = np.array(data, ndmin=1)		

	else:
		input_from_file = False


	# Assure binning factors are not negative
	bin_img = 1 if bin_img < 1 else bin_img
	bin_spec = 1 if bin_spec < 1 else bin_spec


	# Rescale header elements by binning factors
	if input_from_file:
		header['CDELT1'] *= bin_img
		header['CRPIX1'] /= bin_img
		header['CDELT2'] *= bin_img
		header['CRPIX2'] /= bin_img
		if 'CDELT3' in header:
			header['CDELT3'] *= bin_spec
			header['CRPIX3'] /= bin_spec

	# Bin the spectrum and image if required
	if bin_spec > 1 or bin_img > 1:
		print_(f"Original cube shape: {np.shape(data)}", fname, verbose=verbose)

		if data.ndim == 3:
			data = block_reduce(data, [bin_spec,1,1], func=np.nanmean) if bin_spec > 1 else data
			data = block_reduce(data, [1,bin_img,bin_img], func=np.nanmean) if bin_img > 1 else data

		else:
			raise Exception("[!] Input data does not have three dimensions, impossible to detect spectral axis.")

		print_(f"Binned cube shape: {np.shape(data)}", fname, verbose=verbose)


	# Obtain the pixel size in arcseconds. Function argument pixel_size (in arcsec) overrides all other options.
	if pixel_size in [0, None]: 

		# Read from the header
		if input_from_file:
			pixel_size = np.float64(np.abs(header['CDELT1']))*(u.deg).to(u.arcsec)

		# Ask the user to enter it
		else:
			pixel_size = float(input("Enter pixel size in arcsec: "))


	# Convert fwhm to numpy array. If a scalar, use it both as bmin and bmaj.
	fwhm = np.array(fwhm) if np.size(fwhm) > 1 else np.array([fwhm,fwhm])

	# Convert FWHM["] --> FWHM[px]
	fwhm_pix = fwhm / pixel_size	
	fwhm_to_std = np.sqrt(8*np.log(2))
	sigma = fwhm_pix / fwhm_to_std

	# Create the Gaussian Kernel
	kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1], theta=pa*u.deg.to(u.rad)).array

	print_(f'Data shape: {np.shape(data)}', fname, verbose=verbose)	
	print_(f'Kernel shape: {np.shape(kernel)}', fname, verbose=verbose)
	print_(f'Convolving ...', fname, verbose=verbose)

	convolved_data = []

	# Loop over the cube
	for idx, channel in enumerate(data):

		# Use Fast Fourier Transform only for arrays of side-length larger than 500.
		convolver_string = "Using FFT ..." if len(channel) < 500 else "Not using FFT ..."
		if idx == 0: print_(convolver_string, fname, verbose=verbose)

		# Convolve de the image.
		c = convolve(channel, kernel) if len(channel) < 500 else convolve_fft(channel, kernel)

		# Rescale intensity (Jy/px) --> (Jy/beam)
		# Rescaling factor: Output_beam_area / Input_beam_area
		# Output_beam area = Area of a Gassian beam
		# Area of a Gaussian beam: 2*pi/(8*ln(2)) * (FWHM_maj*FWHM_min) = 1.133 * FWHM**2

		rescaling_factor = 1.1331 * (fwhm[0]*fwhm[1]) / (pixel_size**2)

		c = c * rescaling_factor

		# Convert intensities (Jy/beam) to brightness temperature (K)
		Tmb = brightness_temperature(c, unit='Jy/beam', nu=nu, fwhm=fwhm) if output_Tmb	else c

		# Rescale by the telescope efficiency if indicated ()
		if eff != 1: Tmb = (1/eff) * Tmb

		# Flip lef-right if indicated
		if flip_lr:	Tmb = np.fliplr(Tmb)
		
		# Flip upside-down if indicated
		if flip_ud:	Tmb = np.flipud(Tmb)
		
		# Convert to logscale if indicated
		if log_scale: Tmb = np.log10(Tmb)

		convolved_data.append(Tmb)
		

	# Write additional keywords to header
	if input_from_file:
		header['BTYPE'] = 'Tmb' if output_Tmb else 'intensity'
		header['BTYPE'] = 'K' if output_Tmb else 'Jy/beam'
		header['BMIN'] = fwhm[0]*(u.arcsec).to(u.deg)
		header['BMAJ'] = fwhm[1]*(u.arcsec).to(u.deg) 
		header['BPA'] = pa
		header['TELESCOP'] = str(telescope)

	# Write data to fits file if required
	write_fits(outfile, convolved_data, header, overwrite, fname, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return np.array(convolved_data)	 


def add_noise(infile, rms, mean=0, nu=0, outfile='', bin_img=1, bin_spec=1, flip_lr=False, flip_ud=False, log_scale=False, overwrite=False, verbose=False):
	"""
	Function used to add gaussian noise to an image.
	rms must be given in K
	Noise can be added either to single images or cubes.
	"""
	from astropy.nddata.utils import block_reduce

	
	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	# Detects whether input data comes from a file or an array
	if type(infile) == str:
		input_from_file = True
		header = fits.getheader(infile)
		data = fits.getdata(infile)

	elif isinstance(infile, (list, np.ndarray)):
		input_from_file = False
		data = np.array(infile) 
		# Ensure data has at least one dimension
		data = np.array(data, ndmin=1)		

	if bin_img < 1:
		bin_img = 1	
	if bin_spec < 1:
		bin_spec = 1

	# Rescale header elements by binning factors
	if input_from_file:
		header['CDELT1'] *= bin_img
		header['CRPIX1'] /= bin_img
		header['CDELT2'] *= bin_img
		header['CRPIX2'] /= bin_img
		if 'CDELT3' in header:
			header['CDELT3'] *= bin_spec
			header['CRPIX3'] /= bin_spec

	# Bin the spectrum and image if required
	if bin_spec > 1 or bin_img > 1:
		print_(f"Original cube shape: {np.shape(data)}", fname, verbose=verbose)

		if data.ndim == 3:
			# bin spec
			if bin_spec > 1:
				data = block_reduce(data, [bin_spec,1,1], func=np.nanmean)
			if bin_img > 1:
				data = block_reduce(data, [1,bin_img,bin_img], func=np.nanmean)

		else:
			raise Exception("[!] Input data does not have three dimensions, impossible to detect spectral axis.")

		print_(f"Binned cube shape: {np.shape(data)}", fname, verbose=verbose)

	# Validate rms parameter type                                
	if not isinstance(rms, (tuple)):
		raise ValueError('Not a valid type for rms. Must be a tuple (val, "unit").')

	if rms[1].upper() != 'K':
		raise ValueError("rms must be in kelvin. Example: rms=(1,'K').")

	print_("Adding noise level of: %e [K] " % (rms[0]), fname, verbose=verbose)

	noisy_data = []

	print_("Adding Gaussian noise ...", fname, verbose=verbose)

	# Loop over the cube
	for idx, channel in enumerate(data):

		# Add gaussian noise to the image	
		noise = np.random.normal(mean, rms[0], np.shape(channel))
		c = channel + noise

		# Flip image 
		if flip_lr:
			c = np.fliplr(c)
		if flip_ud:
			c = np.flipud(c)
		if log_scale:
			c = np.log10(c)

		noisy_data.append(c)

	# Write data to fits file if required
	write_fits(outfile, noisy_data, header, overwrite, fname, verbose)
		
	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return np.array(noisy_data)	 


def create_moment(infile, outfile='', moment=0, axis=0, d_v=None, min=None, max=None, overwrite=False, verbose=False):
	"""
	Collapse cube into moment maps.
	If the spectral axis comes in frequency and velocity is needed, it will
	rewrite the cube with the converted spectral axis and compute the map 
	from it.
	Based on CASA moment maps
	src: https://casa.nrao.edu/Release3.4.0/docs/userman/UserManse41.html
	"""

	from astropy.nddata.utils import block_reduce
	
	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	# Detects whether input data comes from a file or an array
	if type(infile) == str:
		input_from_file = True
		header = fits.getheader(infile)
		data = fits.getdata(infile)

	elif isinstance(infile, (list, np.ndarray)):
		input_from_file = False
		data = np.array(infile) 
		# Ensure data has at least one dimension
		data = np.array(data, ndmin=1)		


	# Read d_v. If provided as a func. arg.,
	# it overrides values found in the fits header.
	if d_v is not None:
		pass

	elif input_from_file:
		if header['CUNIT3'] == 'km/s':
			d_v = header['CDELT3']
		
		elif header['CUNIT3'] == 'm/s':
			d_v = header['CDELT3']*(u.m/u.s).to(u.km/u.s)

		elif header['CUNIT3'] == 'cm/s':
			d_v = header['CDELT3']*(u.cm/u.s).to(u.km/u.s)
				
	else:
		d_v = float(input("Enter velocity resolution [km/s]: "))		
	print_(f"using d_v={d_v} km/s.", fname, verbose=verbose)	


	# Clip data between min and max if required
	if min != None:
		# data = data.clip(min=min)
		data[data < min] = 0

	if max != None:
		# data = data.clip(max=max)
		data[data > max] = 0


	# Check moment order 
	if moment == -1:
		# Mean value of the spectrum
		data = (1/len(data))*np.nansum(data, axis)

	elif moment == 0:
		# Integrated value of the spectrum
		data = np.nansum(data*d_v, axis)

	elif moment == 1:
		# Intensity weighted coordinate (Velocity field)
		print_("moment=1 not yet implemented.", fname, verbose=verbose)

	# Add necessary kewords
	if input_from_file:
		header['BUNIT'] = 'Jy/beam.km/s'


	# Write data to fits file if required
	write_fits(outfile, data, header, overwrite, fname, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return np.array(data)



def create_datacube(infile, outfile='', b_type='intensity', nu=0, mol='', source_name='', spec_axis=True, overwrite=False, verbose=False):
	"""
	Retrieves data and header from infile and  
	append the data from all velocity maps into one single 3d array (data cube).
	Add necessary keywords to the cube.

	NOTE: Wildcards are allowed by the infile argument. Thanks to glob.
	"""
	from glob import glob

	def del_key(d, k):
		try:
			del d[k]
		except:
			pass

	fname = sys._getframe().f_code.co_name
	start_time = time.time()

	# Loop over input files 
	cube = list()	
	for velmap in sorted(glob(infile)):
		input_from_file = True

		d = fits.getdata(str(velmap))
		header = fits.getheader(str(velmap))

		if 'intensity' in b_type.lower():	
			btype = 'Intensity'
			b_unit = 'Jy/pixel '
			idx = 0			
			cube.append(d[idx])

		elif 'optical_depth' in b_type.lower():
			btype = 'Optical depth'
			b_unit = ''
			idx = 4			
			cube.append(d[idx])

		else:
			raise NameError("Brightness type incorrect. Pick 'intensity' or 'optical_depth'")


	#Write required keywords
	#1st axis
	header['CTYPE1'] = 'RA---SIN'
	header['CRVAL1'] = np.float(header['CRVAL1A'])
	# header['CRPIX1'] = np.float(header['NAXIS1']/2 + 1)
	header['CRPIX1'] = np.float(header['CRPIX1A'])
	header['CDELT1'] = -np.float(header['CDELT1A'])
	header['CUNIT1'] = 'deg'
	header['CROTA1'] = np.float(0)
	#2nd axis
	header['CTYPE2'] = 'DEC--SIN'
	header['CRVAL2'] = np.float(header['CRVAL2A'])
	# header['CRPIX2'] = np.float(header['NAXIS1']/2 + 1)
	header['CRPIX2'] = np.float(header['CRPIX2A'])
	header['CDELT2'] = np.float(header['CDELT2A'])
	header['CUNIT2'] = 'deg'
	header['CROTA2'] = np.float(0)
	#3rd axis
	if spec_axis:
		header['NAXIS'] = 3
		header['CTYPE3'] = 'VRAD'
		header['CRVAL3'] = np.float(0)
		header['CRPIX3'] = np.float(int((header['CHANNELS']/2)+1) if (header['CHANNELS']%2 != 0) else int((header['CHANNELS']/2)))
		header['CDELT3'] = np.float((2*header['MAXVEL'])/(header['CHANNELS']))
		header['CUNIT3'] = 'm/s'
		header['CROTA3'] = np.float(0)

	else:
		header['NAXIS'] = 2
		for k in ['CTYPE3','CRPIX3','CDELT3','CUNIT3','CRVAL3','CROTA3']:
			del_key(header, k)

	# Add missing keywords 	src: http://www.alma.inaf.it/images/ArchiveKeyworkds.pdf
	if source_name != '':
		header['OBJECT'] = source_name
	if mol != '':
		header['MOLECULE'] = mol
	header['BTYPE'] = btype
	header['BSCALE'] = 1.0
	header['BUNIT'] = b_unit
	header['BMAJ'] = np.abs(header['CDELT1'])
	header['BMIN'] = np.abs(header['CDELT2'])
	header['BPA'] = 0
	header['BZERO'] = 0.0
	header['RADESYS'] = 'ICRS'
	header['SPECSYS'] = 'LSRK'
	header['TIMESYS'] = 'UTC'
	header['DATE'] = f'{str(datetime.utcnow()).split()[0]}T{str(datetime.utcnow()).split()[1]}'
	header['FREQ'] = np.float32(nu)
	header['RESTFRQ'] = np.float32(nu)

	# Delete extra WCSs included in the header
	keywords_to_delete = ['CTYPE1A','CRVAL1A','CRPIX1A','CDELT1A','CUNIT1A','CTYPE1B','CRVAL1B','CRPIX1B','CDELT1B','CUNIT1B','CTYPE1C','CRVAL1C','CRPIX1C','CDELT1C','CUNIT1C']
	keywords_to_delete += ['CTYPE2A','CRVAL2A','CRPIX2A','CDELT2A','CUNIT2A','CTYPE2B','CRVAL2B','CRPIX2B','CDELT2B','CUNIT2B','CTYPE2C','CRVAL2C','CRPIX2C','CDELT2C','CUNIT2C']
	keywords_to_delete += ['DETGRID','ID','CRPIX4','CTYPE4','CDELT4','CUNIT4','CRVAL4','NAXIS4']
	
	for k in keywords_to_delete:
		del_key(header, k)


	# Write data to fits file if required
	write_fits(outfile, cube, header, overwrite, fname, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, fname, verbose)

	return np.array(cube)


### END OF FUNCTION DEFINITIONS



if __name__ == "__main__":
	pass
