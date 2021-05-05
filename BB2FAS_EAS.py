from qcore.timeseries import read_ascii, BBSeis
import numpy as np
import pandas as pd
from qcore.shared import exe
import os.path
import jinja2
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
#mpl.style.use('classic')


from argparse import ArgumentParser

G=100 * 9.80665 # cm/s2

asc2smc_ctl="asc2smc.ctl"
smc2fs2_ctl="smc2fs2.ctl"
TEMPLATE=".template"


#FREQ= pd.read_csv("freq.csv",header=None,index_col=None,delim_whitespace=True).T.values.flatten()
FREQ= np.array([1.3182570e-02, 1.3489630e-02, 1.3803850e-02, 1.4125380e-02,
       1.4454400e-02, 1.4791090e-02, 1.5135620e-02, 1.5488170e-02,
       1.5848932e-02, 1.6218101e-02, 1.6595870e-02, 1.6982440e-02,
       1.7378010e-02, 1.7782792e-02, 1.8197010e-02, 1.8620870e-02,
       1.9054604e-02, 1.9498450e-02, 1.9952621e-02, 2.0417380e-02,
       2.0892960e-02, 2.1379621e-02, 2.1877620e-02, 2.2387214e-02,
       2.2908680e-02, 2.3442290e-02, 2.3988330e-02, 2.4547090e-02,
       2.5118870e-02, 2.5703962e-02, 2.6302684e-02, 2.6915352e-02,
       2.7542291e-02, 2.8183832e-02, 2.8840320e-02, 2.9512094e-02,
       3.0199520e-02, 3.0902960e-02, 3.1622774e-02, 3.2359361e-02,
       3.3113110e-02, 3.3884413e-02, 3.4673680e-02, 3.5481333e-02,
       3.6307800e-02, 3.7153530e-02, 3.8018941e-02, 3.8904520e-02,
       3.9810720e-02, 4.0738031e-02, 4.1686940e-02, 4.2657950e-02,
       4.3651580e-02, 4.4668360e-02, 4.5708820e-02, 4.6773523e-02,
       4.7863020e-02, 4.8977890e-02, 5.0118730e-02, 5.1286142e-02,
       5.2480750e-02, 5.3703181e-02, 5.4954090e-02, 5.6234132e-02,
       5.7543992e-02, 5.8884363e-02, 6.0255960e-02, 6.1659500e-02,
       6.3095730e-02, 6.4565412e-02, 6.6069360e-02, 6.7608304e-02,
       6.9183100e-02, 7.0794582e-02, 7.2443600e-02, 7.4131020e-02,
       7.5857751e-02, 7.7624710e-02, 7.9432822e-02, 8.1283050e-02,
       8.3176370e-02, 8.5113793e-02, 8.7096370e-02, 8.9125104e-02,
       9.1201100e-02, 9.3325440e-02, 9.5499262e-02, 9.7723722e-02,
       1.0000000e-01, 1.0232930e-01, 1.0471290e-01, 1.0715192e-01,
       1.0964782e-01, 1.1220184e-01, 1.1481540e-01, 1.1748980e-01,
       1.2022643e-01, 1.2302690e-01, 1.2589254e-01, 1.2882494e-01,
       1.3182570e-01, 1.3489630e-01, 1.3803841e-01, 1.4125373e-01,
       1.4454400e-01, 1.4791083e-01, 1.5135611e-01, 1.5488170e-01,
       1.5848931e-01, 1.6218100e-01, 1.6595870e-01, 1.6982440e-01,
       1.7378010e-01, 1.7782800e-01, 1.8197010e-01, 1.8620871e-01,
       1.9054610e-01, 1.9498443e-01, 1.9952620e-01, 2.0417380e-01,
       2.0892961e-01, 2.1379620e-01, 2.1877620e-01, 2.2387212e-01,
       2.2908680e-01, 2.3442290e-01, 2.3988330e-01, 2.4547090e-01,
       2.5118860e-01, 2.5703954e-01, 2.6302680e-01, 2.6915344e-01,
       2.7542290e-01, 2.8183830e-01, 2.8840312e-01, 2.9512092e-01,
       3.0199520e-01, 3.0902954e-01, 3.1622780e-01, 3.2359364e-01,
       3.3113110e-01, 3.3884412e-01, 3.4673681e-01, 3.5481340e-01,
       3.6307810e-01, 3.7153521e-01, 3.8018940e-01, 3.8904511e-01,
       3.9810714e-01, 4.0738030e-01, 4.1686940e-01, 4.2657953e-01,
       4.3651580e-01, 4.4668360e-01, 4.5708820e-01, 4.6773510e-01,
       4.7863010e-01, 4.8977881e-01, 5.0118720e-01, 5.1286131e-01,
       5.2480750e-01, 5.3703180e-01, 5.4954090e-01, 5.6234130e-01,
       5.7543992e-01, 5.8884360e-01, 6.0255950e-01, 6.1659491e-01,
       6.3095730e-01, 6.4565430e-01, 6.6069340e-01, 6.7608300e-01,
       6.9183093e-01, 7.0794580e-01, 7.2443592e-01, 7.4131023e-01,
       7.5857760e-01, 7.7624710e-01, 7.9432821e-01, 8.1283044e-01,
       8.3176370e-01, 8.5113810e-01, 8.7096360e-01, 8.9125090e-01,
       9.1201080e-01, 9.3325424e-01, 9.5499250e-01, 9.7723710e-01,
       1.0000000e+00, 1.0232930e+00, 1.0471290e+00, 1.0715192e+00,
       1.0964780e+00, 1.1220182e+00, 1.1481534e+00, 1.1748973e+00,
       1.2022641e+00, 1.2302690e+00, 1.2589260e+00, 1.2882500e+00,
       1.3182570e+00, 1.3489630e+00, 1.3803842e+00, 1.4125374e+00,
       1.4454400e+00, 1.4791082e+00, 1.5135610e+00, 1.5488164e+00,
       1.5848930e+00, 1.6218100e+00, 1.6595870e+00, 1.6982440e+00,
       1.7378010e+00, 1.7782794e+00, 1.8197010e+00, 1.8620870e+00,
       1.9054610e+00, 1.9498444e+00, 1.9952621e+00, 2.0417380e+00,
       2.0892960e+00, 2.1379620e+00, 2.1877610e+00, 2.2387211e+00,
       2.2908680e+00, 2.3442290e+00, 2.3988330e+00, 2.4547090e+00,
       2.5118863e+00, 2.5703960e+00, 2.6302680e+00, 2.6915350e+00,
       2.7542283e+00, 2.8183830e+00, 2.8840310e+00, 2.9512090e+00,
       3.0199520e+00, 3.0902960e+00, 3.1622780e+00, 3.2359370e+00,
       3.3113110e+00, 3.3884413e+00, 3.4673681e+00, 3.5481340e+00,
       3.6307800e+00, 3.7153520e+00, 3.8018932e+00, 3.8904510e+00,
       3.9810710e+00, 4.0738030e+00, 4.1686940e+00, 4.2657952e+00,
       4.3651580e+00, 4.4668354e+00, 4.5708813e+00, 4.6773510e+00,
       4.7863001e+00, 4.8977870e+00, 5.0118720e+00, 5.1286130e+00,
       5.2480740e+00, 5.3703184e+00, 5.4954090e+00, 5.6234130e+00,
       5.7543992e+00, 5.8884363e+00, 6.0255960e+00, 6.1659493e+00,
       6.3095730e+00, 6.4565420e+00, 6.6069340e+00, 6.7608284e+00,
       6.9183082e+00, 7.0794563e+00, 7.2443600e+00, 7.4131030e+00,
       7.5857760e+00, 7.7624710e+00, 7.9432821e+00, 8.1283044e+00,
       8.3176364e+00, 8.5113792e+00, 8.7096350e+00, 8.9125070e+00,
       9.1201070e+00, 9.3325410e+00, 9.5499230e+00, 9.7723722e+00,
       1.0000000e+01, 1.0232930e+01, 1.0471284e+01, 1.0715192e+01,
       1.0964780e+01, 1.1220183e+01, 1.1481534e+01, 1.1748973e+01,
       1.2022642e+01, 1.2302684e+01, 1.2589251e+01, 1.2882492e+01,
       1.3182563e+01, 1.3489624e+01, 1.3803840e+01, 1.4125370e+01,
       1.4454392e+01, 1.4791080e+01, 1.5135614e+01, 1.5488170e+01,
       1.5848933e+01, 1.6218101e+01, 1.6595870e+01, 1.6982440e+01,
       1.7378010e+01, 1.7782793e+01, 1.8197010e+01, 1.8620870e+01,
       1.9054610e+01, 1.9498443e+01, 1.9952621e+01, 2.0417380e+01,
       2.0892960e+01, 2.1379620e+01, 2.1877611e+01, 2.2387210e+01,
       2.2908672e+01, 2.3442283e+01, 2.3988321e+01, 2.4547080e+01,
       2.5118860e+01, 2.5703950e+01, 2.6302670e+01, 2.6915340e+01,
       2.7542291e+01, 2.8183832e+01, 2.8840320e+01, 2.9512094e+01,
       3.0199520e+01, 3.0902954e+01, 3.1622780e+01, 3.2359363e+01,
       3.3113110e+01, 3.3884414e+01, 3.4673683e+01, 3.5481334e+01,
       3.6307800e+01, 3.7153514e+01, 3.8018932e+01, 3.8904510e+01,
       3.9810710e+01, 4.0738020e+01, 4.1686930e+01, 4.2657940e+01,
       4.3651570e+01, 4.4668342e+01, 4.5708810e+01, 4.6773500e+01,
       4.7862991e+01, 4.8977890e+01, 5.0118730e+01, 5.1286144e+01,
       5.2480751e+01, 5.3703182e+01, 5.4954090e+01, 5.6234130e+01,
       5.7543991e+01, 5.8884361e+01, 6.0255954e+01, 6.1659500e+01,
       6.3095730e+01, 6.4565414e+01, 6.6069330e+01, 6.7608283e+01,
       6.9183082e+01, 7.0794560e+01, 7.2443572e+01, 7.4131004e+01,
       7.5857734e+01, 7.7624690e+01, 7.9432792e+01, 8.1283030e+01,
       8.3176350e+01, 8.5113770e+01, 8.7096321e+01, 8.9125100e+01,
       9.1201100e+01, 9.3325440e+01, 9.5499260e+01, 9.7723724e+01,
       1.0000000e+02])

def extract_station_acc(bb, statname, outpath):

    times = np.linspace(0, bb.dt * (bb.nt - 1), bb.nt)
    acc = bb.acc(statname).T
    valsEW = acc[0] # 090 -> EW
    valsNS = acc[1] # 000 -> NS


    pd.set_option("precision", 2)
    bbp_df = pd.DataFrame(
        data={
            "#time(sec)": times,
            "N-S(cm/s/s)": valsNS * G,
            "E-W(cm/s/s)": valsEW * G
        },
        dtype=float

    )
    outfile = os.path.join(outpath,statname+".bbp")
    bbp_df.to_csv(outfile, header=bbp_df.columns.values, index=None, sep=" ", float_format='%.6e')
    return outfile



def compute_fas(bbp_file_path, outpath):
    cols = {'NS':2, 'EW':3}


    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)

    fas_files = []
    for comp, col in cols.items():

        template = template_env.get_template(asc2smc_ctl+TEMPLATE)
        extension1= ".smc8.{}".format(comp)
        output_txt = template.render(nheaders = 1, datacolumn = col, extension_string = extension1 , bbp_file_path=bbp_file_path)

        ctl_file_name = asc2smc_ctl+".{}".format(comp)
        with open(ctl_file_name,"w") as f:
            f.writelines(output_txt)
            f.write("\n")

        p=Popen(["./asc2smc"],stdin=PIPE,stdout=PIPE)
        res=p.communicate(bytes(ctl_file_name+"\n",'ascii'))


        template = template_env.get_template(smc2fs2_ctl + TEMPLATE)
        extension2 = ".no_smooth.fs.col"
        to_process_file = os.path.join(outpath,os.path.basename(bbp_file_path)+extension1)
        output_txt = template.render(name_string = extension2, filepath_to_process = to_process_file)

        ctl_file_name = smc2fs2_ctl + ".{}".format(comp)
        with open(ctl_file_name, "w") as f:
            f.writelines(output_txt)
            f.write("\n")

        p=Popen(["./smc2fs2"],stdin=PIPE,stdout=PIPE)
        res=p.communicate(bytes(ctl_file_name+"\n",'ascii'))

        #copy(bbp_file_path+extension1, outpath)
        #os.remove(bbp_file_path+extension1)

        fas_files.append(to_process_file+extension2)

    return fas_files

def compute_eas(bbp_file_path, fas_ns, fas_ew, outpath, plot=True):
    ns_df = pd.read_csv(fas_ns, delim_whitespace=True, header=3, index_col=None)
    ew_df = pd.read_csv(fas_ew, delim_whitespace=True, header=3, index_col=None)
    ns_df = ns_df[ns_df['freq']!=0] #remove freq = 0
    ew_df = ew_df[ew_df['freq']!=0]

    freq=ns_df['freq'].values
    fas_ns=ns_df['fas'].values
    fas_ew=ew_df['fas'].values

    eas = np.sqrt(0.5*(fas_ns**2+fas_ew**2))


    eas_smoothing = k098_smoothing(freq,eas,freq[1]-freq[0],188.5)
    eas_interp = np.exp(np.interp(np.log(FREQ),np.log(freq),np.log(eas_smoothing),left=np.NAN,right=np.NAN))


    pd.set_option("precision", 2)
    eas_df = pd.DataFrame(
        data={
            "#freq": freq,
            "EAS": eas,
            "EAS_smooth": eas_smoothing
        },
        dtype=float

    )

    eas_interp_df = pd.DataFrame(
        data={
            "#freq":FREQ,
            "EAS_interp":eas_interp
        },
        dtype=float
    )

    outfiles = []
    outfile_basename = os.path.basename(bbp_file_path)
    statname  = outfile_basename.split(".")[0]

    outfile = os.path.join(outpath,outfile_basename+".EAS.csv")
    outfiles.append(outfile)
    eas_df.to_csv(outfile, header=eas_df.columns.values, index=None, sep=" ", float_format='%.6e')

    outfile = os.path.join(outpath,outfile_basename+".EAS_interp.csv")
    outfiles.append(outfile)
    eas_interp_df.to_csv(outfile, header=eas_interp_df.columns.values, index=None, sep=" ", float_format='%.6e')


    return outfiles

def plot(fas_ns_csv,fas_ew_csv, eas_csv,eas_interp_csv):
    ns_df = pd.read_csv(fas_ns_csv, delim_whitespace=True, header=3, index_col=None)
    ew_df = pd.read_csv(fas_ew_csv, delim_whitespace=True, header=3, index_col=None)
    ns_df = ns_df[ns_df['freq']!=0] #remove freq = 0
    ew_df = ew_df[ew_df['freq']!=0]

    eas_df = pd.read_csv(eas_csv, header=0,index_col=0,sep=" ")
    eas_interp_df = pd.read_csv(eas_interp_csv, header=0, index_col=0,sep=" ")
    
    freq=ns_df['freq'].values
    fas_ns=ns_df['fas'].values
    fas_ew=ew_df['fas'].values
    eas=eas_df['EAS'].values
    eas_smoothing=eas_df['EAS_smooth'].values
    eas_interp=eas_interp_df['EAS_interp'].values


    statname=os.path.basename(fas_ns_csv).split(".")[0]
    outpath=os.path.abspath(os.path.dirname(fas_ns_csv))

    #plot

    mpl.rcParams['lines.linewidth'] = 0.4
    mpl.rcParams['lines.markersize'] = np.sqrt(2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0.01,100)
    ax.set_ylim(0.00001,1000)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(freq,fas_ns,'bo') #NS
    ax.plot(freq,fas_ew,'r*') #EW
    ax.plot(freq, eas, 'y.')  # EAS

    ax.plot(freq,eas_smoothing, 'k-') #Smoothed EAS
    ax.plot(FREQ,eas_interp,'x',c='#555555') #Smoothed and Interpolated

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Fourier Amplitude (cm/s)')
    ax.legend(labels=('NS','EW','EAS','Smoothed EAS','Smoothed and Interpolated'),loc='lower left')
    ax.set_title('FAS/EAS: {}'.format(statname))
    #plt.show()
    plt.savefig(os.path.join(outpath,statname+".png"),dpi=500)
    plt.close(fig)



def k098_smoothing(freq,y,dx,bexp):
    # # ** smoothing of a function y (equally-spaced, dx) with the "Konno-Ohmachi"
    # # ** function sin (alog10(f/fc)^exp) / alog10(f/fc)^exp) ^^4
    # # ** where fc is the frequency around which the smoothing is performed
    # # ** exp determines the exponent 10^(1/exp) is the half-width of the peak
    # # ** cf Konno & Ohmachi, 1998, BSSA 88-1, pp. 228-241
    #
    # bexp is "b" from the PEER RVT report definition
    # b = 2 * pi / bw
    # bw = 2 * pi / b
    # SP(boore) = 4 / b
    #
    # bexp = 188.5 corresponds to bw = 0.033, SP = 0.02122(not a lot of smoothing, PEER RVT value)
    # bexp = 40    corresponds to bw = 0.157, SP = 0.1(much more smoothing)
    # bexp = 20    corresponds to bw = 0.314, SP = 0.2(even more smoothing, what Boore's notes suggest)

    nx = freq.size
    ysmooth = np.zeros(y.size)
    fratio = np.power(10,2.5/bexp)
    ysmooth[0]=y[0]

    for ix in range(1,nx):
        fc=freq[ix]
        fc1 = fc/fratio
        fc2 = fc*fratio
        ix1 = int (np.floor(fc1/dx))-1 #be careful here...
        ix2 = int (np.floor(fc2/dx +1))-1
        if ix1 < 1:
            ix1 = 0
        if ix2 >= nx-1:
            ix2 = nx-1
        a1 = 0
        a2 = 0

        for jj in range(ix1,ix2+1):
            if jj != ix:
                c1 = bexp * np.log10(freq[jj]/fc)
                c1 = np.power((np.sin(c1)/c1),4)
                a2 = a2 + c1
                a1 = a1 + c1 * y[jj]
            else:
                a2 = a2 + 1
                a1 = a1 + y[ix]
            ysmooth[ix] = a1 / a2

    return ysmooth

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("bb_path", help="Give a full path to BB.bin")
    parser.add_argument("statname", help="Station name to compute FAS and EAS")
    parser.add_argument("outpath", help="Give a full path to output files")

    args = parser.parse_args()

    bb = BBSeis(args.bb_path)
    inputfile=extract_station_acc(bb,args.statname,args.outpath)
#    inputfile="InputAcc/10000000.8001-CLS.acc.bbp"
    fas_files = compute_fas(inputfile, args.outpath)
    eas_files = compute_eas(inputfile,fas_files[0],fas_files[1],args.outpath)

    plot(fas_files[0],fas_files[1],eas_files[0],eas_files[1])
 
