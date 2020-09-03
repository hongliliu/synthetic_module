from toolkit import *

def pipeline(core, run, mol, d, obs, createDatacube=False, applyBeam=False, columnDensity=False, addNoise=False, callCASA=False, overwrite=False, verbose=False):
	"""
		Run the pipeline.
		Enable/disable steps by setting the boolean arguments passed True/False, respectively.
	"""
	pipeline_time = time.time()
	
	try:
		if True in [createDatacube, applyBeam, columnDensity, addNoise]:			

			if createDatacube:
				# Arrange data into datacubes	
				print("[pipeline] creating datacube... ")
				create_datacube(f'velmaps/vel_channel_maps_species_{mol.molID}_line_0001_vel_????.fits',\
								 outfile=f'{mol.name}_intensity.fits', b_type='intensity',\
								 mol=mol.name, \
								 nu=mol.nu, \
								 source_name=core.name, \
								 overwrite=overwrite, \
								 verbose=verbose \
								)
							
			if applyBeam:			
				# Convolve cubes with a gaussian PSF	
				print("[pipeline] Convolving... ")
				apply_beam(	f'{mol.name}_intensity.fits', \
							outfile=f'{mol.name}_intensity_{obs}.fits', \
							bin_img=10, \
							bin_spec=67, \
							fwhm=mol.fwhm, \
							overwrite=overwrite, \
							verbose=verbose \
							)

			if addNoise:
				# Add noise to the convolved cubes
				add_noise(	f'{mol.name}_intensity_{obs}.fits', \
							outfile=f'{mol.name}_intensity_{obs}_noisy.fits', \
							rms=2.48e-3, \
							rms_unit='K',\
							mean=0, \
							nu=mol.nu, \
							theta=mol.fwhm, \
							flip_lr=False, \
							flip_ud=False, \
							log_scale=False, \
							overwrite=overwrite, \
							verbose=verbose \
							)

			if columnDensity:				
				# Compute column densities 	
				print("[pipeline] Computing column densities... ")
				column_density(f'{mol.name}_intensity_datacube_{obs}_noisy.fits', \
								outfile=f'{mol.name}_column_density_{obs}_noisy.fits', \
								mol=mol.name, \
								nu=mol.nu, \
								Tex=mol.Tex, \
								mu=mol.mu, \
								J=mol.J, \
								E_J1=mol.E_J1, \
								E=mol.E, \
								g=mol.g, \
								A=mol.A, \
								Q=mol.Q, \
								overwrite=overwrite, \
								verbose=verbose \
								)

		else:
			raise IOError(f'{color.BOLD}[pipeline]{color.ENDC} No task selected.')
			sys.exit(1)

	except IOError as error:
		print(error)

	if callCASA:
		print(f'{color.BOLD}[pipeline]{color.ENDC} Calling CASA script... ')
		os.system('casa --nogui --nocrashreport -c alma_simulation.py ')				

	return 0


if __name__ == "__main__":
	"""
		Execution of the pipeline.
		The following will be executed when the script itself is called from the command line.
	"""	
		
	obs = 'alma'
	pipe_time = time.time()

	for core in [Lmu10M2()]:

		for mol in [oH2Dp()]:

				for d in [f'd{i}kpc' for i in range(1,8)]:

					for run in core.runs:

						print(f'{color.BOLD}\n[pipeline]{color.ENDC} Running pipeline for {core.name} at {d} for {run} for {mol.name}')
	
						status = pipeline(core, run=run, mol=mol, obs=obs, d=d, createDatacube=True, applyBeam=True, addNoise=True, columnDensity=True, callCASA=False, overwrite=True, verbose=True )
						
						if status == 0:
							print(f'{color.BOLD}[pipeline]{color.ENDC} Pipeline finished succefully. Elapsed time: {time.strftime("%H:%M:%S",time.gmtime(time.time()-pipe_time))}')

						else:	
							print(f'{color.BOLD}[pipeline]{color.ENDC} Pipeline aborted. Elapsed time: {time.strftime("%H:%M:%S",time.gmtime(time.time()-pipe_time))}')
