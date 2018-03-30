import numpy as np
import ephem
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import pandas as pd
import scipy as sp
import scipy.signal as sig
import scipy.interpolate as interp
from pytz import timezone

def get_sec(dtime):
	return dtime.hour * 3600 + dtime.minute * 60 + dtime.second

def get_hour(dtime):
	return get_sec(dtime)/3600

def temp2hr(temp):
	return temp*24/100

def smooth(data, Nfilt=101):
	N = len(data)
	fitdat = np.tile(data,3)
	filt = sig.savgol_filter(fitdat, Nfilt, 3)
	out = np.array([max(0,x) for x in filt[N:2*N]])
	return out

def plot_daynight(ax, theta, sunrise, sunset, roffset, alpha=0.2):
	top = np.ones_like(theta)*24 + roffset
	bottom = np.ones_like(theta)*0+roffset
	ax.fill_between(theta, bottom, np.concatenate((sunrise,np.reshape(sunrise[0], (1,))))+roffset, facecolor='black', alpha=alpha, zorder=1.5, lw=0.)
	ax.fill_between(theta, np.concatenate((sunset,np.reshape(sunset[0], (1,))))+roffset, top, facecolor='black', alpha=alpha, zorder=1.5, lw=0.)

def plot_equinoxes(ax, theta, roffset, style='k-',lw=0.5,alpha=0.5,):
	dth = theta[1]-theta[0]
	seq = (datetime.date(2017,3,20)-datetime.date(2017,1,1)).days
	ssol = (datetime.date(2017,6,20)-datetime.date(2017,1,1)).days
	feq = (datetime.date(2017,9,22)-datetime.date(2017,1,1)).days
	wsol = (datetime.date(2017,12,21)-datetime.date(2017,1,1)).days
	wstyle = 'b-'
	sstyle = 'r-'

	ax.plot(np.array([dth, dth])*seq+theta[0], np.array([0,24])+roffset,style,lw=lw,alpha=alpha)
	ax.plot(np.array([dth, dth])*ssol+theta[0], np.array([0,24])+roffset,sstyle,lw=lw,alpha=alpha)
	ax.plot(np.array([dth, dth])*feq+theta[0], np.array([0,24])+roffset,style,lw=lw,alpha=alpha)
	ax.plot(np.array([dth, dth])*wsol+theta[0], np.array([0,24])+roffset,wstyle,lw=lw,alpha=alpha)

def plot_moon(ax, theta, moonrise, moonset, moonphase, roffset):
	for i in range(min(len(moonrise),len(moonset))):
		if moonrise[i] < moonset[i]:
			start = moonrise[i]+roffset
			end = moonset[i]+roffset
			ax.plot([theta[i], theta[i]], [start,end],'k', alpha=moonphase[i])
		else:
			start=0+roffset
			end = moonset[i]+roffset
			ax.plot([theta[i], theta[i]], [start,end],'k', alpha=moonphase[i])
			start = moonrise[i]+roffset
			end = 24+roffset
			ax.plot([theta[i], theta[i]], [start,end],'k', alpha=moonphase[i])

def plot_line(ax, theta, data, roffset, color):
	toplt = np.concatenate((data,np.reshape(data[0], (1,))))
	ax.plot(theta, toplt+roffset, color, linewidth=4)

def filter_rise(lin):    
	for i in range(1,len(lin)):
		try:
			if lin[i] == lin[i-1]:
				lin.pop(i)
		except:
			pass
	for i in range(1,len(lin)):
		try:
			if lin[i] < lin[i-1]:
				lin.insert(i,0.)
		except:
			pass

def filter_set(lin):    
	for i in range(1,len(lin)):
		try:
			if lin[i] == lin[i-1]:
				lin.pop(i)
		except:
			pass
	for i in range(1,len(lin)):
		try:
			if lin[i] < lin[i-1]:
				lin.insert(i,0.)
		except:
			pass

def convert_to_abs_coords(array_in, roffset):
	out = np.empty_like(array_in, dtype=float)
	out[0,:] = array_in[0,:]
	for i in range(1,out.shape[0]):
		out[i,:] = out[i-1,:]+array_in[i,:]
	out[:,0] = out[:,0]-758
	out[:,1] = out[:,1]-100
	ymax = 2210
	xmax = 6014
	out[:,1] = (ymax-out[:,1])/ymax*24 + roffset
	out[:,0] = out[:,0]/xmax*2*np.pi
	return out

def make_extra_minr(minth, maxth, roffset, N=300):
	th = np.linspace(minth, maxth, N)
	extra_min = np.zeros((N,2))+roffset
	extra_min[:,0] = th
	return extra_min

def make_extra_maxr(minth, maxth, roffset, N=300):
	extra_max = make_extra_minr(minth, maxth, roffset, N=N)
	extra_max[:,1] += 24
	return extra_max

def make_extra_minth(roffset,N=4):
	r = np.linspace(roffset, roffset+24, N)
	extra_min = np.zeros((N,2))
	extra_min[:,1] = r
	return extra_min

def make_extra_maxth(roffset,N=4):
	r = np.linspace(roffset, roffset+24, N)
	extra_max = np.ones((N,2))*2*np.pi
	extra_max[:,1] = r
	return extra_max

def shift_theta(patchdata, dth):
	tmp = np.array(patchdata)
	tmp[:,0] += dth
	return tmp

def make_tmap():
	Tdict = {120:1.,
			 110:0.95,
			 100:0.85,
			 90:0.8,
			 80:0.7,
			 70:0.65,
			 60:0.5,
			 50:0.38,
			 40:0.25,
			 30:0.18,
			 20:0.1,
			 10:0.05,
			 0:0.0}

	r,g,b = [],[],[]
	rtup,gtup,btup = [],[],[]
	Tsorted = [(k,v) for k,v in Tdict.items()]
	Tsorted.sort()
	for t,v in Tsorted:
		tmp = mpl.cm.nipy_spectral(v)
		r = tmp[0]
		g = tmp[1]
		b = tmp[2]
		tn = t/120
		rtup.append((tn,r,r))
		gtup.append((tn,g,g))
		btup.append((tn,b,b))
	cdict = {'red':rtup,'green':gtup,'blue':btup}
	tmap = mpl.colors.LinearSegmentedColormap('tempmap',cdict)
	return tmap


def make_smap(max_alpha=0.5):
	N =10
	rtup = []
	gtup = []
	btup = []
	atup = []
	tmp = mpl.cm.Purples(1.)
	r = tmp[0]
	g = tmp[1]
	b = tmp[2]
	for x in np.linspace(0,1,N):
		tmp = mpl.cm.Purples(x)
		# r = tmp[0]
		# g = tmp[1]
		# b = tmp[2]
		a = x*max_alpha
		rtup.append((x,r,r))
		gtup.append((x,g,g))
		btup.append((x,b,b))
		atup.append((x,a,a))
		cdict = {'red':rtup,'green':gtup,'blue':btup, 'alpha':atup}
		cmap = mpl.colors.LinearSegmentedColormap('tempmap',cdict)
	return cmap
	
def make_pmap():
	N = 10
	rtup = []
	gtup = []
	btup = []
	atup = []
	for x in np.linspace(0,1,N):
		tmp = mpl.cm.Blues(x)
		r = tmp[0]
		g = tmp[1]
		b = tmp[2]
		rtup.append((x,r,r))
		gtup.append((x,g,g))
		btup.append((x,b,b))
		atup.append((x,x,x))
		# cdict = {'red':rtup,'green':gtup,'blue':btup, 'alpha':atup}
		cdict = {'red':rtup,'green':gtup,'blue':btup}
		cmap = mpl.colors.LinearSegmentedColormap('tempmap',cdict)
	return cmap


def get_average_hourly_temp(file, timezone_name):
	df = pd.read_table(file, 
					   delimiter='\s+', 
					   skipinitialspace=True, 
					   na_values=['*','**','***','****','*****','******'], 
					   parse_dates=['YR--MODAHRMN'], 
					   index_col='YR--MODAHRMN',
					  usecols=['YR--MODAHRMN','TEMP'])
	df.dropna(axis=1, how='all', inplace=True)
	
	utc = timezone('UTC').localize(df.index)
	df['hour'] = utc.astimezone(timezone(timezone_name)).hour
	df['dayofyear'] = df.index.dayofyear
	dfday = df.groupby(['dayofyear','hour'])
	dyhrtemp = dfday['TEMP'].mean()
	temps = dyhrtemp.unstack().values
	return df, temps

def smooth_temps(temps,s=1.6e5, plot=False):
	days = np.linspace(0,366,366)
	hours = np.linspace(0,24,366)
	tbiv = interp.RectBivariateSpline(range(24),range(366),temps.T,s=0)
	tinterp = tbiv(hours, days)
	tbiv2 = interp.RectBivariateSpline(days, days,tinterp, s=s)
	tsmooth = tbiv2(days,days)
	if plot:
		tmap = make_tmap()
		plt.figure(figsize=(16,4))
		plt.subplot(121)
		times = np.linspace(0,24,24)
		days = np.linspace(0,365,366)
		plt.pcolormesh(days, times, temps.T,vmin=0, vmax=120,cmap=tmap)
		plt.title('Temperature')
		plt.ylabel('hour')
		plt.xlabel('day of year')
		plt.xlim(0,365)
		plt.ylim(0,24)
		plt.colorbar()
		plt.grid(False)

		plt.subplot(122)
		times = np.linspace(0,24,366)
		plt.pcolormesh(days, times, tsmooth,vmin=0, vmax=120,cmap=tmap)
		plt.title('Smoothed Temperature')
		plt.ylabel('hour')
		plt.xlabel('day of year')
		plt.xlim(0,365)
		plt.ylim(0,24)
		plt.colorbar()
		plt.grid(False)
	return tsmooth

def get_localhour_from_ephemtime(ephemtime, timezone_name):
	dt_utc = timezone('UTC').localize(ephem.date(ephemtime).datetime())
	dt_local = dt_utc.astimezone(timezone(timezone_name))
	return dt_local.hour + dt_local.minute/60 + dt_local.second/3600

def get_sunriseset(lat, lon, elevation, timezone_name, start_date='2017-01-01 00:01'):
	o = ephem.Observer()
	o.lat=lat
	o.lon=lon
	o.elevation=elevation
	o.date=start_date

	s=ephem.Sun()
	s.compute(o)

	sunrise = []
	sunset = []

	for i in range(365):
		s.compute(o)

		srise = get_localhour_from_ephemtime(o.next_rising(s), timezone_name)
		sset = get_localhour_from_ephemtime(o.next_setting(s), timezone_name)

		sunrise.append(srise)
		sunset.append(sset)
		o.date +=1
	return np.array(sunrise), np.array(sunset)

def read_SGHDtemps_to_dataframe(filename, timezone_name):
	df = pd.read_table(filename, 
					   delimiter='\s+', 
					   skipinitialspace=True, 
					   na_values=['*','**','***','****','*****','******'], 
					   parse_dates=['YR--MODAHRMN'], 
					   index_col='YR--MODAHRMN',
					  usecols=['YR--MODAHRMN','TEMP'])
	df.dropna(axis=1, how='all', inplace=True)
	df.index = df.index.tz_localize('UTC')
	df.index = df.index.tz_convert(timezone_name)
	df['hour'] = df.index.hour
	df['dayofyear'] = df.index.dayofyear
	return df

def convert_SGHDdf_to_avghourlytemp(df):
	dfday = df.groupby(['dayofyear','hour'])
	dyhrtemp = dfday['TEMP'].mean()
	temps = dyhrtemp.unstack().values
	return temps

def store_hourlytemps_as_csv(temps, filename):
	np.savetxt(filename,temps,delimiter=',')

def load_hourlytemps_computed(filename):
	data = np.loadtxt(filename,delimiter=',')
	return data

def compute_month_doy():
	monthstarts = []
	for i in range(1,13):
		monthstarts.append((datetime.date(2017, i, 1) - datetime.date(2017, 1, 1)).days)
	monthstarts = np.array(monthstarts)
	return monthstarts

def import_daily_normal_precip(filename):
	dn = pd.read_csv(filename)
	precip = np.zeros((365))
	precip[0] = dn['YTD-PRCP-NORMAL'].values[0]
	precip[1:] = np.diff(dn['YTD-PRCP-NORMAL'].values)
	return precip

def import_daily_precip_computed(filename):
	dn = pd.read_csv(filename)
	return dn['precip'].values, dn['snow'].values

def smooth_precip(precip,s=0.2e-1,dx=20,plot=False):
	y = np.tile(precip,3)
	x = range(365*3)
	begin = 365-dx
	end = 365*2+dx
	f = interp.UnivariateSpline(x[begin:end],y[begin:end],s=s,k=5,bbox=[None,None])
	precip_sm = f(x[365:365*2])
	precip_sm[precip_sm<0.] = 0.
	if plot:    	
		plt.plot(precip,'.',alpha=0.2)
		plt.plot(precip_sm,zorder=2.1,lw=2,alpha=1)
		plt.title('smoothed precipitation')
		plt.ylabel('daily rain (in)')
		plt.xlabel('day of year')
		plt.xlim(0,365)
		plt.ylim(ymin=-0.01)
		plt.grid()
	return precip_sm

def plot_tempychart(precip=None, temps=None, sunrise=None, sunset=None, title=None, snow = None, savename=None, roffset=10, equinoxes=False):
	if savename is None:
		savename = title+'.png'
	dth = 2*np.pi/len(sunrise)
	start = np.pi/2
	theta = -np.linspace(0-start, 2*np.pi-start, len(sunrise)+1)
	msize = 4
	roffset = 10

	monthstarts = compute_month_doy()

	fig = plt.figure(figsize=(14,14))
	ax = plt.subplot(111, projection='polar')
	ax.fill_between(theta, 0,roffset,facecolor='white')

	# temperature
	tmap = make_tmap()
	times = np.linspace(0,24,366)+roffset
	ax.pcolormesh(theta,times,temps, vmin=0,vmax=120,cmap=tmap)

	# equinox/solstice
	if equinoxes:
		plot_equinoxes(ax, theta, roffset, alpha=0.5, lw=0.7)

	# sunrise and sunset
	plot_daynight(ax, theta, sunrise, sunset, roffset, alpha=0.25)

	# moonrise/set and phase
	# plot_moon(ax, theta, moonrise, moonset, moonphase, roffset)

	# precipitation
	if precip is not None:
		p2d = np.tile(precip[:,None],2)
		pmap = make_pmap()
		plt.pcolormesh(theta,[0,roffset],np.concatenate((p2d,p2d[0,:][None,:]),axis=0).T,cmap=pmap,vmax=0.3,vmin=0.)

	if snow is not None:
		smap = make_smap()
		s2d = np.tile(snow[:,None],2)
		plt.pcolormesh(theta,[0,roffset],np.concatenate((s2d,s2d[0,:][None,:]),axis=0).T,cmap=smap,vmax=.75,vmin=0.)

	# axes and labels
	ax.plot(theta, np.ones_like(theta)*roffset,'k-')
	ax.set_rmax(24+roffset)
	ax.set_xticks(theta[monthstarts])
	ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], )
	ax.set_yticklabels([])  
	ax.grid(False)
	ax.text(0,0,title,fontsize=20,horizontalalignment='center',verticalalignment='center', color='black')
	plt.savefig(savename)
	plt.close()