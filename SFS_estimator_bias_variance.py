"""
runs an analysis of estimator bias and variance .  
For a range of true values of the selection coefficient g generate boxplots of estimates 
usage: SFS_estimator_bias_variance.py [-h] [-c FIX_THETA_RATIO] [-b] [-d DENSITYOF2NS] -f FOLDSTATUS [-i] [-k NTRIALS] [-l PLOTFILELABEL] [-m MAX2NS] -nc NC [-q THETAS] [-s SEED] [-t THETAN] [-w] [-x GDENSITYMAX] [-D] [-F CSFSPREFIX]
                                      [-G NSEQS] [-L SEQLEN] [-M MAXI] [-N POPSIZE] [-O OUTPUT_DIR] [-P] [-R REC] [-S] [-U MU] [-W PARENT_DIR] [-slim] [-model MODEL]

options:
  -h, --help          show this help message and exit
  -c FIX_THETA_RATIO  set the fixed value of thetaS/thetaN
  -b                  run the basinhopping optimizer after the regular optimizer
  -d DENSITYOF2NS     gamma or lognormal, only if simulating a distribution of Ns, else single values of Ns are used
  -f FOLDSTATUS       usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded'
  -i                  run the optimizer 3 times, not just once
  -k NTRIALS          number of trials per parameter set
  -l PLOTFILELABEL    optional string for labelling plot file names
  -m MAX2NS           optional setting for the maximum 2Ns
  -nc NC              # of sampled chromosomes i.e. 2*(# diploid individuals)
  -q THETAS           theta for selected sites
  -s SEED             random number seed (positive integer)
  -t THETAN           set theta for neutral sites, optional when -r is used, if -t is not specified then thetaN and thetaS are given by -q
  -w                  do not estimate both thetas, just the ratio
  -x GDENSITYMAX      maximum value of 2Ns density, default is 1.0, use with -d
  -D                  debug
  -F CSFSPREFIX       optional prefix for csfs filenames, e.g. Afr, Eur or EAs
  -G NSEQS            Number of sequences
  -L SEQLEN           Sequence length
  -M MAXI             optional setting for the maximum bin index to include in the calculations
  -N POPSIZE          Population census size
  -O OUTPUT_DIR       Path for output directory
  -P                  load previously generated slim files, use with -W
  -R REC              Per site recombination rate per generation
  -S                  Save simulated SFS to a file
  -U MU               Per site mutation rate per generation
  -W PARENT_DIR       Path for working directory
  -slim               simulate SFSs with SLiM
  -model MODEL        The demographic model to simulate SFSs
"""

import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from  scipy.optimize import basinhopping
from scipy.stats import chi2
import SFS_functions
import math
import argparse
import simulate_SFS_withSLiM as slim
import os
import os.path as op
import random
import time
# Add the directory containing your module to sys.path
sys.path.append('/mnt/d/genemod/better_dNdS_models/popgen/manuscript/figs')
import twoDboxplot 

# import warnings
# warnings.filterwarnings("ignore")

# as of 8/23/2023 Kuethe method does not work as well

# optimizemethod="Powell"  #this just does not seem to do very well 
# also tried BFGS using a calculated jacobian,  but it was incredibly slow and ran into nan errors 
optimizemethod="Nelder-Mead" # this works best but not perfect,  take the best of 3 calls 
np.seterr(divide='ignore', invalid='ignore')

DEBUGMODE = False 

def makeSFScomparisonstring(headers,sfslist):
    slist = []
    slist.append("\t".join(headers) + "\n")
    n = len(sfslist[0])
    k = len(sfslist)
    for i in range(n):
        if k ==6:
            temp = ["{}".format(sfslist[0][i]),"{}".format(sfslist[1][i]),"{:.3g}".format(sfslist[2][i]),"{}".format(sfslist[3][i]),"{}".format(sfslist[4][i]),"{:.3g}".format(sfslist[5][i])]
        else:
            temp = ["{}".format(sfslist[j][i]) for j in range(k)]
        temp.insert(0,str(i))
        slist.append("\t\t" + "\t".join(temp)+"\n")
    slist.append("\n")   
    return ''.join(slist) 

def make_outfile_name_base(args):
    if args.plotfilelabel != "" and args.plotfilelabel[-1] != "_": # add a spacer if needed
        args.plotfilelabel += "_"  
    a = ["{}ratioPRF_k{}_n{}_Qs{:.0f}_Qn{:.0f}_".format(args.plotfilelabel,args.ntrials,args.nc,args.thetaS,args.thetaN)]
    if args.foldstatus:
        a.append("{}_".format(args.foldstatus))
    if args.densityof2Ns != "single2Ns":
        a.append("{}_".format(args.densityof2Ns))
    if args.gdensitymax != 1.0:
        a.append("gmax{}_".format(args.gdensitymax))
    if args.use_theta_ratio:
        a.append("WQn_".format(args.densityof2Ns))      
    if args.use_theta_ratio:
        if args.fix_theta_ratio:    
            a.append("Wfx_")
        else:
            a.append("WQn_")      
    if args.maxi:
        a.append("maxi{}_".format(args.maxi))
    basename = ''.join(a)
    if basename[-1] =='_':
        basename = basename[:-1]
    # print (args)
    # print(basename)
    basename = op.join(args.output_dir,basename)
    return basename

def extract_float(s):
    """ 
    to get the float in a string that looks like this: # 4Nmu(exon total length)=1036.8 distribution=lognormal dist_pars=[0.3, 0.5] n=100 Selected folded SFS
    """
    parts = s.split()  # Split the string into parts separated by spaces
    for part in parts:
        if "=" in part: # and part[0].isdigit():
            # Split at '=' and take the second part, then convert to float
            num_part = part.split('=')[1]
            try:
                return float(num_part)
            except ValueError:
                pass  # Ignore if the conversion fails and continue
    return None  # Return None if no float is found


def getslimgeneratedratios(args,parent_dir,nc, foldstatus,densityof2Ns,maxi = None):
    """
        gets sfss that were previously generated by simulate_SFS_withSLiM.py running SLiM
    """
    if DEBUGMODE:
        if densityof2Ns =="lognormal":
            gvalstrs = [["0.3","0.5"]]
        elif densityof2Ns == "gamma":
            gvalstrs = [["11.0","0.1"]]
        elif densityof2Ns  == "single2Ns":
            gvalstrs = [["-50.0"]]            

    else:

        if densityof2Ns == "lognormal":
            # gvalstrs = [["0.3","0.5"], ["1.0","0.7"], ["2.0","1.0"], ["2.2","1.4"], ["3.0","1.2"]]
            gvalstrs = [["0.3","0.5"], ["1.0","0.7"], ["2.0","1.0"], ["3.0","1.2"]]
        elif densityof2Ns == "gamma":
            gvalstrs = [["11.0","0.1"],["8.5","0.2"],["3.86","0.7"],["4.64","1.1"]]
        elif densityof2Ns == "normal":
            exit()
        elif densityof2Ns == "single2Ns":
            gvalstrs = [["-50.0"],["-10.0"],["-5.0"],["-1.0"],["0.0"],["1.0"],["5.0"],["10.0"],["50.0"]]
    
    gvals = [list(map(float,temp)) for temp in gvalstrs]
    minNval = 0
    numdatasets = len(gvalstrs)
    ssfslists = [[] for i in range(numdatasets)]
    nsfslists = [[] for i in range(numdatasets)]
    ratiolists = [[] for i in range(numdatasets)]
    truethetaNlist = [[] for i in range(numdatasets)]
    truethetaSlist = [[]for i in range(numdatasets)]
    
    # for di,d in enumerate(gdirs):
    for di,g in enumerate(gvalstrs):
        if densityof2Ns in ("normal","lognormal","gamma","discrete3"):
            tempdir = op.join(parent_dir,"{}-{}".format(g[0],g[1]))
        else:
            tempdir = op.join(parent_dir,"{}".format(g[0]))
        if args.csfsprefix is not None:
            sfile = op.join(tempdir,args.csfsprefix + "_csfs_selected.txt")
            nfile = op.join(tempdir,args.csfsprefix + "_csfs_neutral.txt")
        else:
            sfile = op.join(tempdir,"csfs_selected.txt")
            nfile = op.join(tempdir,"csfs_neutral.txt")
        nlines = open(nfile,'r').readlines() 
        truethetaNlist[di].append(extract_float(nlines[0]))
        # 4Nmu(exon total length)=1036.8 distribution=lognormal dist_pars=[0.3, 0.5] n=100 Selected folded SFS
        nsfss = nlines[1:]
        slines = open(sfile,'r').readlines() 
        truethetaSlist[di].append(extract_float(slines[0]))
        ssfss = slines[1:]
        numdatarows = len(nsfss)
        assert len(ssfss)==numdatarows
        for i in range(numdatarows):
            neutvals = list(map(float,nsfss[i].strip().split()))
            selectvals = list(map(float,ssfss[i].strip().split()))
            assert len(neutvals)==len(selectvals)
            nbins = len(selectvals) 
            if foldstatus=="isfolded":
                assert nbins == 1+nc//2,"# of sfs bins {} not consistent with -f {} and reported sample size -n {}".format(nbins,foldstatus,nc)
            if foldstatus in ("unfolded", "foldit"):
                assert nbins == 1+nc,"# of sfs bins {} not consistent with -f {} and reported sample size -n {}".format(nbins,foldstatus,nc)
                neutvals=neutvals[:-1]
                selectvals = selectvals[:-1]
                if foldstatus == "foldit":
                    selectvals = [0] + [selectvals[j]+selectvals[nc-j] for j in range(1,nc//2)] + [selectvals[nc//2]]
                    neutvals = [0] + [neutvals[j]+neutvals[nc-j] for j in range(1,nc//2)] + [neutvals[nc//2]]
            if maxi and maxi < len(selectvals):
                selectvals = selectvals[:maxi+1]
                neutvals = neutvals[:maxi+1]
            ssfslists[di].append(selectvals)
            nsfslists[di].append(neutvals)
            ratiolists[di].append([math.inf if neutvals[j] <= minNval else selectvals[j]/neutvals[j] for j in range(len(neutvals))])
            assert len(ratiolists[di])==len(nsfslists[di])
    
    return numdatarows,gvalstrs,ssfslists,nsfslists,ratiolists,truethetaNlist,truethetaSlist

def boundarycheck(x, bounds):
    check = False
    for i,val in enumerate(x):
        check = check or math.isclose(val,bounds[i][0])
        check = check or math.isclose(val,bounds[i][1])
        if check:
            break
    return check

def run(args):
    global DEBUGMODE 
    if args.DEBUGMODE:
        DEBUGMODE = True 

    starttime = time.time()
    SFS_functions.SSFconstant_dokuethe = False #  True

    np.random.seed(args.seed)
    ntrialsperg = args.ntrials
    densityof2Ns = args.densityof2Ns 
    foldstatus = args.foldstatus
    usebasinhopping = args.usebasinhopping
    use_theta_ratio = args.use_theta_ratio
    fix_theta_ratio = args.fix_theta_ratio
    maxi = args.maxi
    estimatemax2Ns = True if  args.max2Ns is None and args.densityof2Ns != "single2Ns" and args.densityof2Ns != "normal" else False
    fixedmax2Ns = args.max2Ns
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dofoldedlikelihood = foldstatus in ('isfolded','foldit')
    gdm = args.gdensitymax
    if gdm != 1.0:
        SFS_functions.reset_g_xvals(gdm)

    nc = args.nc

    # Additional parameters for SLiM
    simuslim = args.simuslim
    if simuslim:
        model = args.model
        mu = args.mu
        rec = args.rec
        popSize = args.popSize
        seqLen = args.seqLen
        nSeqs = args.nSeqs
        parent_dir = args.parent_dir
        savefile = args.savefile
    loadsfsfiles =  args.loadsfsfiles
    if loadsfsfiles:
        parent_dir = args.parent_dir
        model = args.model

    doWFPRFsimulations = not (args.simuslim or args.loadsfsfiles)
    if doWFPRFsimulations:
        if args.thetaN:
            thetaN = args.thetaN
            thetaNresults = []
        else:
            thetaN = args.thetaS
            args.thetaN = args.thetaS
        thetaS = args.thetaS
    
    if loadsfsfiles:
        ntrialsperg,gvals,ssfslists,nsfslists,ratiolists,truethetaNlist,truethetaSlist = getslimgeneratedratios(args,parent_dir,nc,foldstatus,densityof2Ns,maxi=args.maxi)
        if hasattr(args, "ntrials") and args.ntrials is not None and args.ntrials < ntrialsperg:
            ntrialsperg = args.ntrials
            # print(ntrialsperg)
        if args.plotfilelabel == "":
            basename = op.join(args.output_dir,parent_dir.strip(".").replace("/","_").strip("_"))
        else:
            basename = op.join(args.output_dir,args.plotfilelabel + parent_dir.strip(".").replace("/","_"))
    else:
        basename = make_outfile_name_base(args)
        if densityof2Ns == 'lognormal':  
            gvals = [[0.3, 0.5], [1, 0.7], [2.0, 1.0], [2.2, 1.4], [3, 1.2]]
        elif densityof2Ns == 'gamma':
            gvals = [[3, 2],[5.5, 1.5],[2.1, 0.1],[4, 0.5],[6, 1]]
            # [[11, 0.1], [8.5, 0.2], [3.86, 0.7], [4.64, 1.1]]
        elif densityof2Ns == "single2Ns":
            gvals = [-50,-10,-5,-1]        
            # gvals = [-10,-5,-2,-1,0,1,2,5,10]        
    print(basename)
    if densityof2Ns != "single2Ns":
        ln1results = []
        ln2results = []
        allgresults = []
    else:
        gresults = []

    foldstring = "folded" if foldstatus else "unfolded"
    SFScomparesultsstrings = []
    
    if densityof2Ns != "single2Ns":
        plotfile1name = '{}_term1_plot.png'.format(basename)
        plotfile2name = '{}_term2_plot.png'.format(basename)
        plotfile2Dname = '{}_2Dplot.png'.format(basename)
        gvalsfor2Dplotsnames = "{}_gvals_for_2D_plot.txt".format(basename)
        
    else:
        plotfilename = '{}_2Ns_plot.png'.format(basename)
        gvalsforplotnames = "{}_gvals_for_plot.txt".format(basename)
    estimates_filename = '{}_results.txt'.format(basename)    
    

    thetaNresults = []
    thetaSresults = []
    thetaratioresults = []
    savedSFSS = [[] for i in range(len(gvals))]  
    savedSFSN = [[] for i in range(len(gvals))]     
    savedDataRatios = [[] for i in range(len(gvals))]
    if estimatemax2Ns:
        max2Nsresults = []
    if use_theta_ratio:
        func = SFS_functions.NegL_SFSRATIO_estimate_thetaratio
        argtuple = (nc,dofoldedlikelihood,densityof2Ns,fix_theta_ratio,fixedmax2Ns,False,False)
    else:
        func = SFS_functions.NegL_SFSRATIO_estimate_thetaS_thetaN
        argtuple = (nc,dofoldedlikelihood,densityof2Ns,False,fixedmax2Ns,False,False)
    if use_theta_ratio and fix_theta_ratio is None:
        thetaratioresults = []
    for gi,g in enumerate(gvals):
        thetaSresults.append([])
        thetaNresults.append([])
        if use_theta_ratio:
            thetaratioresults.append([])
        if estimatemax2Ns:
            max2Nsresults.append([])
        if densityof2Ns in ("normal","lognormal","gamma","discrete3"):
            ln1results.append([])
            ln2results.append([])
            allgresults.append([[],[]])
        else:
            gresults.append([])
        for i in range(ntrialsperg):
            if densityof2Ns in ("normal","lognormal","gamma","discrete3"):
                SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1],i+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
            else:
                SFScompareheaders = ["2Ns:{} Trial#{}:".format(g,i+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
            if simuslim:
                if densityof2Ns in ("normal","lognormal","gamma","discrete3"):
                    nsfs, ssfs, ratios, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                nsdist = densityof2Ns, nsdistargs = g, diploidsamplesize  = int(nc/2), model = model, 
                                                                                nSeqs = nSeqs, parent_dir = parent_dir, savefile = savefile)
                else:
                    nsfs, ssfs, ratios, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                nsdist = 'fixed', ns = g, diploidsamplesize  = int(nc/2), model = model, 
                                                                                nSeqs = nSeqs, parent_dir = parent_dir, savefile = savefile)

            elif loadsfsfiles:
                nsfs = nsfslists[gi][i]
                ssfs = ssfslists[gi][i]
                ratios = ratiolists[gi][i]
            else:
                nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,gdm,nc,args.maxi,dofoldedlikelihood,densityof2Ns,[g],None,False,None) 

            temp = list(argtuple)
            temp.append(ratios)
            arglist = tuple(temp)
            thetaNest = sum(nsfs)/sum([1/i for i in range(1,nc)]) # this should work whether or not the sfs is folded 
            thetaSest = sum(ssfs)/sum([1/i for i in range(1,nc)]) # this should work whether or not the sfs is folded 
            thetaratioest = thetaSest/thetaNest
            bounds = []
            startarray = []
            if use_theta_ratio:
                if fix_theta_ratio is None:
                    bounds.append((thetaratioest/20,thetaratioest*20))
            else:
                bounds += [(thetaNest/30,thetaNest*30),(thetaSest/30,thetaSest*30)]
                # startarray += [thetaNest,thetaSest]
            if densityof2Ns == "single2Ns": #single 2Ns value 
                bounds += [(min(-100,-abs(float(g[0]))*10),max(100,abs(float(g[0]))*10))]
            elif densityof2Ns == "lognormal":
                bounds += [(-5,10),(0.00001,10)]
            elif densityof2Ns =="gamma":
                bounds += [(0.00001,10),(0.00001,10)]
            elif densityof2Ns == "normal":
                bounds += [(-20,20),(0.001,10)]
            elif densityof2Ns == "discrete3":
                bpimds += [(0,1),(0,1)]
            else:
                print("error")
                exit()
            if estimatemax2Ns:
                bounds += [(-20.0,20.0)]
            # do ntries optimzation attempts and pick the best one 
            ntries = 5 if args.optimize5times and DEBUGMODE == False else 1
            rxvals = []
            rfunvals = []
            # print(bounds)
            for ii in range(ntries):
                boundsarray = [
                    (random.uniform(bounds[i][0],bounds[i][0] + 0.3*(bounds[i][1]-bounds[i][0])),
                        random.uniform(bounds[i][1] - 0.3*(bounds[i][1]-bounds[i][0]),bounds[i][1]) ) for i in range(len(bounds))]
                startarray = [(boundsarray[i][0] + boundsarray[i][1])/2.0 for i in range(len(boundsarray))]
                # print(startarray)
                result = minimize(func,np.array(startarray),args=arglist,method=optimizemethod,bounds=bounds)                         
                rxvals.append(result)
                rfunvals.append(-result.fun)
                if DEBUGMODE:
                    if densityof2Ns != "single2Ns":
                        print("{} {} #{} >{} {:.5f} {}".format(g[0],g[1],i,ii,-result.fun,result.x))
                    else:
                        print("{} #{} >{} {:.5f} {}".format(g,i,ii,-result.fun,result.x))
            besti = rfunvals.index(max(rfunvals))
            result = rxvals[besti]
            atboundary = boundarycheck(result.x, bounds)
            if usebasinhopping or (atboundary and DEBUGMODE==False):
                # use result.x to set bounds and start search
                boundsarray = [(v/3,v*3) for v in result.x ]
                startarray = [random.uniform(v/1.5,v*1.5) for v in result.x ] #won't work if v is 0 
                # print(args.parent_dir,result.x,"\n",boundsarray,"\n",startarray)
                bresult = basinhopping(func,np.array(startarray),T=10.0,
                                minimizer_kwargs={"method":optimizemethod,"bounds":boundsarray,"args":arglist})
                if -bresult.fun > -result.fun:
                    result = bresult
            pi = 0
            if densityof2Ns != "single2Ns":
                if use_theta_ratio:
                    if fix_theta_ratio: 
                        tempratio = fix_theta_ratio
                    else:
                        tempratio = result.x[pi]
                        pi+= 1
                    thetaratioresults[gi].append(tempratio)
                    allgresults[gi][0].append(result.x[pi])
                    allgresults[gi][1].append(result.x[pi+1])
                    ln1results[gi].append(result.x[pi])
                    ln2results[gi].append(result.x[pi+1])
                    thetaNresults[gi].append(thetaNest)
                    thetaSresults[gi].append(tempratio*thetaNest)
                    pi+= 2
                    if estimatemax2Ns:
                        max2Nsresults[gi].append(result.x[pi])
                        pi+=1
                else:
                    allgresults[gi][0].append(result.x[2])
                    allgresults[gi][1].append(result.x[3])
                    ln1results[gi].append(result.x[2])
                    ln2results[gi].append(result.x[3])
                    thetaNresults[gi].append(result.x[0])
                    thetaSresults[gi].append(result.x[1])
                    tempratio = None
                    if estimatemax2Ns:
                        max2Nsresults[gi].append(result.x[-1])
                fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],gdm,nc,args.maxi,dofoldedlikelihood,densityof2Ns,[ln1results[gi][-1],ln2results[gi][-1]],None,True,tempratio)
            else:
                if use_theta_ratio:
                    if fix_theta_ratio: 
                        tempratio = fix_theta_ratio
                        thetaratioresults[gi].append(tempratio)
                    else:
                        tempratio = result.x[pi]
                        pi+= 1
                    thetaratioresults[gi].append(tempratio)
                    gresults[gi].append(result.x[pi])
                    thetaNresults[gi].append(thetaNest)
                    thetaSresults[gi].append(tempratio*thetaNest)
                    fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],gdm,nc,args.maxi,dofoldedlikelihood,densityof2Ns,[gresults[gi][-1]],None,True,tempratio)
                else:
                    gresults[gi].append(result.x[2])
                    thetaNresults[gi].append(result.x[0])
                    thetaSresults[gi].append(result.x[1])
                    fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],gdm,nc,args.maxi,dofoldedlikelihood,densityof2Ns,[gresults[gi][-1]],None,True,None)
            SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[nsfs,ssfs,ratios,fitnsfs,fitssfs,fitratios]))
            savedSFSS[gi].append(ssfs)
            savedSFSN[gi].append(nsfs)
            savedDataRatios[gi].append(ratios)
        f = open(estimates_filename,"w")
        f.write("Program SFS_estimator_bias_variance.py results:\n\nCommand line arguments:\n=======================\n")
        for key, value in vars(args).items():
            f.write("\t{}: {}\n".format(key,value))
        f.write("\nCompare simulated and fitted SFS:\n=================================\n")
        f.write(''.join(SFScomparesultsstrings))

        # f = open(estimates_filename,"a")
        f.write("Parameter Estimates:\n===================\n")
        for gi,g in enumerate(gvals):
            if densityof2Ns != "single2Ns":
                if use_theta_ratio:
                    if estimatemax2Ns:  # as of 1/8/2024 all simulations use max2Ns = 1.0
                        if truethetaNlist[gi][0] == None or truethetaSlist[gi][0]==None:
                            f.write("\tSet {} Values:\t\tThetaRatio UNKNOWN\t\tg1 {}\t\tg2 {}\t\tmax2Ns {}\n".format(gi+1,g[0],g[1],1.0))
                        else:
                            f.write("\tSet {} Values:\t\tThetaRatio {}\t\tg1 {}\t\tg2 {}\t\tmax2Ns {}\n".format(gi+1,truethetaSlist[gi][0]/truethetaNlist[gi][0],g[0],g[1],1.0))
                        for k in range(ntrialsperg):
                            f.write("\t\t{}\t\t{:.4g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(k+1,thetaratioresults[gi][k],ln1results[gi][k],ln2results[gi][k],max2Nsresults[gi][k])) 
                        f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(np.mean(np.array(thetaratioresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi])),np.mean(np.array(max2Nsresults[gi]))))
                        f.write("\tStDev:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(np.std(np.array(thetaratioresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi])),np.std(np.array(max2Nsresults[gi]))))
                    else:
                        if truethetaNlist[gi][0] == None or truethetaSlist[gi][0]==None:
                            f.write("\tSet {} Values:\t\tThetaRatio UNKNOWN\t\tg1 {}\t\tg2 {}\n".format(gi+1,g[0],g[1]))
                        else:
                            f.write("\tSet {} Values:\t\tThetaRatio {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,truethetaSlist[gi][0]/truethetaNlist[gi][0],g[0],g[1]))
                        for k in range(ntrialsperg):
                            f.write("\t\t{}\t\t{:.4g}\t\t{:.2g}\t\t{:.2g}\n".format(k+1,thetaratioresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
                        f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.mean(np.array(thetaratioresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
                        f.write("\tStDev:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.std(np.array(thetaratioresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
                else:
                    if estimatemax2Ns: # as of 1/8/2024 all simulations use max2Ns = 1.0
                        if truethetaNlist[gi][0] == None or truethetaSlist[gi][0]==None:
                            f.write("\tSet {} Values:\t\tThetaS UNKNOWN\t\tThetaN UNKNOWN\t\tg1 {}\t\tg2 {}\t\tmax2Ns {}\n".format(gi+1,g[0],g[1],1.0)) # for truetheta vals all true values are same in a set, se we can use the 0th value
                        else:
                            f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg1 {}\t\tg2 {}\t\tmax2Ns\n".format(gi+1,truethetaSlist[gi][0],truethetaNlist[gi][0],g[0],g[1],1.0)) # for truetheta vals all true values are same in a set, se we can use the 0th value
                        for k in range(ntrialsperg):
                            f.write("\t\t{}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],ln1results[gi][k],ln2results[gi][k],ln2results[gi][k],max2Nsresults[gi][k]))
                        f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi])),np.mean(np.array(max2Nsresults[gi]))))
                        f.write("\tStDev:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.4g}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi])),np.std(np.array(max2Nsresults[gi]))))
                    else:
                        if truethetaNlist[gi][0] == None or truethetaSlist[gi][0]==None:                        
                            f.write("\tSet {} Values:\t\tThetaS UNKNOWN\t\tThetaN UNKNOWN\t\tg1 {}\t\tg2 {}\n".format(gi+1,g[0],g[1])) # for truetheta vals all true values are same in a set, se we can use the 0th value
                        else:
                            f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,truethetaSlist[gi][0],truethetaNlist[gi][0],g[0],g[1])) # for truetheta vals all true values are same in a set, se we can use the 0th value
                        for k in range(ntrialsperg):
                            f.write("\t\t{}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
                        f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
                        f.write("\tStDev:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
            
            else:
                if use_theta_ratio:
                    f.write("\tSet {} Values:\t\tThetaRatio {}\t\tg {}\n".format(gi+1,truethetaSlist[gi][0]/truethetaNlist[gi][0],g))
                    for k in range(ntrialsperg):
                        f.write("\t\t{}\t\t{:.4g}\t\t{:.2g}\n".format(k+1,thetaratioresults[gi][k],gresults[gi][k]))
                    f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\n".format(np.mean(np.array(thetaratioresults[gi])),np.mean(np.array(gresults[gi]))))
                    f.write("\tStDev:\t\t{:.3g}\t\t{:.2g}\n".format(np.std(np.array(thetaratioresults[gi])),np.std(np.array(gresults[gi]))))
                else:
                    f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg {}\n".format(gi+1,args.thetaS,args.thetaN,g))
                    for k in range(ntrialsperg):
                        f.write("\t\t{}\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],gresults[gi][k]))
                    f.write("\tMean:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(gresults[gi]))))
                    f.write("\tStDev:\t\t{:.2g}\t\t{:.2g}\t\t{:.2g}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(gresults[gi]))))
        f.write("\n\n")
        f.write("Mean SFS Counts For Each Parameter Set (Neutral, Selected, Ratio):\n==================================================================\n")     
        f.write("Selection parameter sets: {}\n".format(" ".join(list(map(str,gvals)))))
        for gi in range(len(gvals)):   
            f.write("\tNeutral\tSelect\tRatio")
        f.write("\n")
        for i in range(len(savedSFSS[0][0])):
            f.write("{}".format(i))
            for gi in range(len(gvals)):
                ntemp = sum([savedSFSN[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                stemp = sum([savedSFSS[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                rtemp = sum([savedDataRatios[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                f.write("\t{:.2g}\t{:.2g}\t{:.3g}".format(ntemp,stemp,rtemp))
            f.write("\n")
        endtime = time.time()
        total_seconds = endtime-starttime
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        f.write(f"\nTime taken: {hours} hours, {minutes} minutes, {seconds:.2g} seconds\n")            
        f.close()        
        if densityof2Ns != "single2Ns":
            gf = open(gvalsfor2Dplotsnames,'w')
            for gi,g in enumerate(gvals):
                gf.write("{}\t".format(g[0]))
                for i in range(len(allgresults[gi][0])):
                    gf.write("{} ".format(allgresults[gi][0][i]))
                gf.write("\n")
                gf.write("{}\t".format(g[1]))
                for i in range(len(allgresults[gi][1])):
                    gf.write("{} ".format(allgresults[gi][1][i]))
                gf.write("\n\n")                
            gf.close()
            twoDboxplot.make2Dboxplot(allgresults,gvals,"{} term 1".format(densityof2Ns,nc,foldstring),"{} term 2".format(densityof2Ns,nc,foldstring),plotfile2Dname, False)
            term1vals = []
            term2vals = []
            for gi,g in enumerate(gvals):
                term1vals.append(float(g[0]))
                term2vals.append(float(g[1]))
            fig, ax = plt.subplots()
            ax.boxplot(ln1results,showmeans=True,sym='',positions=term1vals)
            plt.xlabel("{} term 1".format(densityof2Ns,nc,foldstring))
            plt.ylabel("Estimates")
            # plt.plot(gvals,gvals)
            plt.plot(term1vals,term1vals)
            plt.savefig(plotfile1name)
            plt.clf
            fig, ax = plt.subplots()
            ax.boxplot(ln2results,showmeans=True,sym='',positions=term2vals)
            plt.xlabel("{} term 2".format(densityof2Ns,nc,foldstring))
            plt.ylabel("Estimates")
            # plt.plot(gvals,gvals)
            plt.plot(term2vals,term2vals)
            plt.savefig(plotfile2name)
            plt.clf
        else:
            gf = open(gvalsforplotnames,'w')
            for gi,g in enumerate(gvals):
                gf.write("{}\t".format(g[0]))
                for i in range(len(gresults[gi])):
                    gf.write("{} ".format(gresults[gi][i]))
                gf.write("\n\n")                
            gf.close()            
            tempgvals = []
            for gi,g in enumerate(gvals):
                tempgvals.append(float(g[0]))
            fig, ax = plt.subplots()
            ax.boxplot(gresults,showmeans=True,sym='',positions=tempgvals)
            plt.xlabel("g (selection)")
            plt.ylabel("Estimates")
            plt.plot(tempgvals,tempgvals)
            plt.savefig(plotfilename)
            plt.clf            
    print("done {}".format(basename))    


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="fix_theta_ratio",default=None,type=float,help="set the fixed value of thetaS/thetaN")
    parser.add_argument("-b",dest="usebasinhopping",action="store_true",default=False,help="run the basinhopping optimizer after the regular optimizer")
    parser.add_argument("-d",dest="densityof2Ns",default = "single2Ns",type=str,help="gamma or lognormal, only if simulating a distribution of Ns, else single values of Ns are used")
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-i",dest="optimize5times",action="store_true",default=False,help="run the optimizer 3 times, not just once ")
    parser.add_argument("-k",dest="ntrials",default = 20, type = int, help="number of trials per parameter set")    
    parser.add_argument("-l",dest="plotfilelabel",default = "", type=str, help="optional string for labelling plot file names ")    
    parser.add_argument("-m",dest="max2Ns",default=None,type=float,help="optional setting for the maximum 2Ns ")
    parser.add_argument("-nc",dest="nc",type = int, required=True,help="# of sampled chromosomes  i.e. 2*(# diploid individuals)  ")
    parser.add_argument("-q",dest="thetaS",type=float,help = "theta for selected sites")    
    parser.add_argument("-s",dest="seed",type = int,help = " random number seed (positive integer)",default=1)
    parser.add_argument("-t",dest="thetaN",type=float,help = "set theta for neutral sites, optional when -r is used, if -t is not specified then thetaN and thetaS are given by -q")
    parser.add_argument("-w",dest="use_theta_ratio",action="store_true",default=False,help="do not estimate both thetas, just the ratio")  
    parser.add_argument("-x",dest="gdensitymax",type=float,default=1.0,help = "maximum value of 2Ns density,  default is 1.0,  use with -d")
    parser.add_argument("-D",dest="DEBUGMODE",action="store_true",default=False,help="debug")
    parser.add_argument("-F", dest="csfsprefix",default = None,type = str, help="optional prefix for csfs filenames,  e.g. Afr, Eur or EAs")
    parser.add_argument("-G", dest="nSeqs", type=int, help="Number of sequences")
    parser.add_argument("-L", dest="seqLen", default=10000, type=int, help="Sequence length")    
    parser.add_argument("-M",dest="maxi",default=None,type=int,help="optional setting for the maximum bin index to include in the calculations")
    parser.add_argument("-N", dest="popSize", default=1000, type=int, help="Population census size")    
    parser.add_argument("-O", dest="output_dir",default = ".",type = str, help="Path for output directory")
    parser.add_argument("-P", dest="loadsfsfiles", action="store_true",default=False,help="load previously generated slim files, use with -W")    
    parser.add_argument("-R", dest="rec", type=float, default=1e-6/4, help="Per site recombination rate per generation")    
    parser.add_argument("-S", dest="savefile",action="store_true", default=False,help="Save simulated SFS to a file")
    parser.add_argument("-U", dest="mu", type=float, default=1e-6/4, help="Per site mutation rate per generation")    
    parser.add_argument("-W", dest="parent_dir",default = "results/slim",type = str, help="Path for working directory")
    parser.add_argument("-slim",dest="simuslim",action="store_true",default=False, help="simulate SFSs with SLiM")
    parser.add_argument("-model",dest="model", type=str, default= "constant", help="The demographic model to simulate SFSs")
        
    args  =  parser.parse_args(sys.argv[1:])  
    if args.simuslim and args.loadsfsfiles:
        parser.error("-slim and -P conflict")    
    if not args.loadsfsfiles:
        if not hasattr(args, "ntrials"):
            parser.error("-k, # trials, required unless -P")
        if not hasattr(args, "thetaS"):
            parser.error("-q, thetaS, required unless -P")            
        if not hasattr(args,"nc"):
            parser.error("-nc required unless -P")
    
    if args.densityof2Ns != "single2Ns" and not (args.densityof2Ns in ['lognormal','gamma']):
        parser.error("-d term {} is not either 'lognormal' or 'gamma'".format(args.densityof2Ns))
    if args.densityof2Ns== "single2Ns" and args.gdensitymax != 1.0:
        parser.error("-x requires that a density function be specified (i.e. -d )")    
    if args.foldstatus not in ("isfolded","foldit","unfolded"):
        parser.error("-f {} is wrong,  this tag requires one of 'isfolded','foldit', or 'unfolded'".format(args.foldstatus))
    if args.usebasinhopping:
        args.optimize5times = True
    args.commandstring = " ".join(sys.argv[1:])
    return args

    # return parser

if __name__ == '__main__':
    """

    """
    starttime = time.time()
    args = parsecommandline()
    run(args)
