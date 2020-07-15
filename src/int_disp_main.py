from pdPythonLib import *
# constants
# assume that intrinsic Q is not dispersive
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time
from scipy import interpolate
import pandas as pd
import h5py
import pdb
import cmath
from numpy import fft
from tqdm import tqdm
import argparse
import matplotlib

# constants
um = 1e-6
c = 299792458
λ_p = 1.55e-6 # pump wavelength
ω_p = 2*np.pi*c/λ_p
f_p = c/λ_p
R = 100
mode_ring = 'tm'
mode_ring_n = 0
Nmodes = 4
field_res = 300
x_size = 6.5
y_size = 5
extraction_gap = 0
w = 4.5
hbar = 6.634e-34/2/np.pi
dint_step = 10

def get_neff(f,Nmodes):
	tm_index = 0
	te_index = 0
	tefrac = []
	neff_te = {}
	neff_tm = {}
	mode_index = {}
	mode_index['TE'] = []
	mode_index['TM'] = []
	f.Exec(f"section_ring.evlist.update")
	for i in range(Nmodes):
		tefrac.append(f.Exec(f"section_ring.evlist.list[{i+1}].modedata.tefrac"))
		if tefrac[i] > 50:
			neff_te[te_index] = np.real(f.Exec(f"section_ring.evlist.list[{i+1}].neff"))
			mode_index["TE"].append(i+1)
			te_index += 1
		else:
			neff_tm[tm_index] = np.real(f.Exec(f"section_ring.evlist.list[{i+1}].neff"))
			mode_index["TM"].append(i+1)
			tm_index += 1
	return tefrac,neff_te,neff_tm,mode_index

def get_ng(f,mode_index,mode="TM",mode_n=0):
	index = mode_index[mode][mode_n]
	f.Exec(f"section_ring.evlist.list[{index}].modedata.update(1)")
	return f.Exec(f"section_ring.evlist.list[{index}].modedata.neffg")

def int_dispersion(λ_p,R,mode_ring,mode_ring_n,Nmodes,field_res,x_size,y_size,w,h,pts,λ_start,λ_end):
	
	um = 1e-6
	c = 299792458
	λ_p = λ_p*um
	ω_p = 2*np.pi*c/λ_p
	f_p = c/λ_p
	extraction_gap = 0
	hbar = 6.634e-34/2/np.pi
	# set up environment
	f = pdApp()
	f.ConnectToApp()

	section_ring = 'Ref& section_ring = app.findnode("/int_dispersion/ring")'
	section_bus = 'Ref& section_bus = app.findnode("/int_dispersion/bus")'
	variables = 'Ref& variables = app.findnode("/int_dispersion/variables")'

	f.Exec(section_ring)
	f.Exec(section_bus)
	f.Exec(variables)
	f.Exec(f"variables.setvariable(N,{Nmodes})")
	f.Exec(f"variables.setvariable(resolution,{field_res})")
	f.Exec(f"variables.setvariable(a,{x_size})")
	f.Exec(f"variables.setvariable(b,{y_size})")
	f.Exec(f"variables.setvariable(lambda_ir,{λ_p*1e6})")
	f.Exec(f"variables.setvariable(radius,{R})")
	f.Exec(f"variables.setvariable(extract_gap,{extraction_gap})")
	f.Exec(f"variables.setvariable(w,{w})")
	f.Exec(f"variables.setvariable(h,{h})")

	# allocate resources and predict first cavity mode frequency
	tefrac,neff_te,neff_tm,mode_index = get_neff(f,Nmodes)
	ng = get_ng(f,mode_index)
	fsr_λ = -1*(λ_p*1e6)**2/(2*np.pi*ng*R)
	fsr_f = c*1e6/(2*np.pi*ng*R)
	print('fsr =',fsr_f*1e-9,'GHz')
	f_start = c/(λ_end*um)
	f_end = c/(λ_start*um)
	m_end = (f_end-f_p)//fsr_f
	m_start = (f_start-f_p)//fsr_f
	print(ng,fsr_λ)
	m_arr = np.linspace(m_start,m_end,pts)
	mdiff = np.diff(m_arr)[0]
	f0_guess = m_arr[0]*fsr_f + f_p
	progress = tqdm(total=len(m_arr)-1)
	freal_arr = np.zeros(len(m_arr))
	λreal_arr = np.zeros(len(m_arr))
	ng_arr = np.zeros(len(m_arr)-1)
	f_curr = f0_guess
	freal_arr[0] += f_curr

	# find cavity mode resonances for each m number
	for i,m in enumerate(m_arr[0:-1]): # don't need the ng of the last point because ng for i to predict f of i+1
		f.Exec(f"variables.setvariable(lambda_ir,{c*1e6/f_curr})")
		f.Exec(f"variables.setvariable(N,{Nmodes})")
		tefrac,neff_te,neff_tm,mode_index = get_neff(f,Nmodes)
		ng = get_ng(f,mode_index,mode_ring,mode_ring_n)
		ng_arr[i] += ng
		f_curr += c*1e6*mdiff/(2*np.pi*ng*R)
		freal_arr[i+1] += f_curr
		progress.update(1)
	progress.close()

	# post data processing and calculated Dint
	pump_idx = np.searchsorted(freal_arr,f_p)
	real_m_arr = np.linspace(-1*(pump_idx),len(m_arr)-pump_idx-1,len(m_arr))*mdiff
	fsr_p = c*1e6/(ng_arr[pump_idx]*2*np.pi*R)
	f_predicted_arr = real_m_arr*fsr_p + freal_arr[pump_idx]
	λ_predicted_arr = c*1e6/f_predicted_arr
	Dint_arr = freal_arr - f_predicted_arr

	return λ_predicted_arr,Dint_arr

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# modes parameters
	parser.add_argument('-wl',type=float,default=1.55,help='pump wavelength in um, default 1.55um')
	parser.add_argument('-radius',type=float,default=100,help='radius of the ring in um, default 50')
	parser.add_argument('-rm_n',type=int,default=0,help='mode number inside ring, default of fundamental mode 0')
	parser.add_argument('-rm',type=str,default='TM',help='mode inside ring, can either be "TE" or "TM", default is "TM"')
	parser.add_argument('-nmodes',type=int,default=10,help='number of modes for FIMMWAVE to simulate, default is 10')
	# environment parameters
	parser.add_argument('-field_res',type=int,default=300,help='x and y resolution of the simulation environment, default 300')
	parser.add_argument('-x_size',type=float,default=6.5,help='x size of the simulation environment in um, default 6.5')
	parser.add_argument('-y_size',type=float,default=4,help='y size of the simulation environment in um, default 4')
	# geometry parameters
	parser.add_argument('-width',type=float,default=3,help='width of waveguide in um, default 3')
	parser.add_argument('-height',type=float,default=1,help='height of waveguide in um, default 1')
	parser.add_argument('-points',type=int,default=10,help='number of points')

	# plot settings
	parser.add_argument('-plot_freq',type=int,default=0,help='plot in frequency domain, default 0 (no), 1 for yes')
	parser.add_argument('-save_to',type=str,default=None,help="Use following style: file.csv")

	parser.add_argument('-start_wl',type=float,default=1,help='starting wavelength for the Dint curve')
	parser.add_argument('-end_wl',type=float,default=2,help="final wavelength for the Dint curve")

	c = 299792458
	um = 1e-6
	args = parser.parse_args()
	λ_arr,Dint_arr = int_dispersion(args.wl,args.radius,args.rm,args.rm_n,
		args.nmodes,args.field_res,args.x_size,args.y_size,args.width,args.height,args.points,args.start_wl,args.end_wl)

	matplotlib.style.use('seaborn-whitegrid')
	fig,ax = plt.subplots(1,1)
	figsize=(5,4)
	fig.set_size_inches(figsize)
	if args.plot_freq:
		ax.plot(c/λ_arr/um,Dint_arr)
		ax.set_xlabel('Frequency (Hz)')
	else:
		ax.plot(λ_arr,Dint_arr)
		ax.set_xlabel('Pump Wavelength (μm)')
	ax.set_title('Integrated Dispersion')
	ax.set_ylabel('$\mathrm{D_{int}}$ (Hz)')
	plt.show()

	filename_head = 'F:\\Josh\\int_dispersion\\'
	if args.save_to is not None:
		df = pd.DataFrame(np.transpose([λ_arr,Dint_arr]))
		df.columns = ['lambda','Dint']
		df.to_csv(filename_head+args.save_to)