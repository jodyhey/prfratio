"""
    reads a file with SFSs 
    runs estimators
usage: run_one_pair_of_SFSs.py [-h] -a PARENT_DIR [-c FIX_THETA_RATIO] [-d DENSITYOF2NS] -f FOLDSTATUS [-g GLOBALOPT] [-i OPTIMIZETIMES] [-l OUTFILELABEL] [-m MAX2NS] [-M MAXI] [-o] [-w] [-z]

options:
  -h, --help          show this help message and exit
  -a PARENT_DIR       Path for SFS file
  -c FIX_THETA_RATIO  set the fixed value of thetaS/thetaN
  -d DENSITYOF2NS     gamma or lognormal or normal, only if simulating a distribution of Ns, else single values of 2Ns are used
  -f FOLDSTATUS       usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded'
  -g GLOBALOPT        b for use basinhopping (best but slow), s for use shgo algorithm, default is neither
  -i OPTIMIZETIMES    run the minimize optimizer # times
  -l OUTFILELABEL     string for labelling resutls file
  -m MAX2NS           optional setting for the maximum 2Ns, use only with -d lognormal or -d gamma
  -M MAXI             optional setting for the maximum bin index to include in the calculations
  -o                  fix the mode of 2Ns density at 0, only works for lognormal and gamma
  -w                  do not estimate both thetas, just the ratio
  -z                  include a proportion of the mass at zero in the density model, requires normal, lognormal or gamma
"""
import numpy as np
from scipy.optimize import minimize,minimize_scalar

from  scipy.optimize import basinhopping, brentq
from scipy.optimize import shgo
import scipy.special as sc
import os.path as op
import  SFS_functions 
import math
import random
import time
import argparse
import sys
import warnings
import traceback

# A custom warning handler
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    file = open("errlog.txt",'a')
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    # print(sys.stderr)

warnings.showwarning = warn_with_traceback

random.seed(1)
sc.seterr(all='raise')
starttime = time.time()
def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if random.random() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number

def getsfss(fn,isfolded = False):
    """
        neutral is first 
        then selected
        must be the same length
        if isfolded nc is calculated assuming a diploid sample
    """
    digits = "0123456789"
    lines = open(fn,'r').readlines()
    sfsindices = [i for i, s in enumerate(lines) if s and s[0].isdigit()]
    numsfss = len(sfsindices)
    sfss = []
    for i in range(numsfss):
        line = lines[sfsindices[i]]
        if "." in line:
            vals = list(map(float,line.strip().split()))
            sfs = list(map(stochastic_round,vals))
        else:
            sfs = list(map(int,line.strip().split()))
        sfs[0] = 0
        sfss.append(sfs)
    if isfolded == False:
        fsfss = []
        nc  = len(sfss[0])
        for sfs in sfss:
            fsfss.append([0] + [10*sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [10*sfs[nc //2]])
    else:
        fsfss = sfss
        sfss = None
        nc = 2*(len(fsfss[0]) - 1)
    return nc,numsfss,sfss, fsfss

# def checkmodestart(densityof2Ns,use_theta_ratio,startarray,bounds):
#     if use_theta_ratio:
#         pi = 1
#         if densityof2Ns=="lognormal":
#             while True:
#                 tempmu = random.uniform(bounds[pi][0],bounds[pi][1])
#                 tempsigma = random.uniform(bounds[pi+1][0],bounds[pi+1][1])
#                 if (tempmu - pow(tempsigma,2)) > 0.0:
#                     startarray[pi] = tempmu
#                     startarray[pi+1] = tempsigma
#                     return startarray
#                     break

#     return startarray
            
def update_table(X, headers, new_data, new_labels):
    # Update headers
    headers.extend(new_labels)
    
    # Format and add new data
    for i in range(len(new_data[0])):
        if len(X) <= i:
            X.append([])
        formatted_row = [f"{new_data[0][i]:.1f}", f"{new_data[1][i]:.1f}", f"{new_data[2][i]:.3f}"]
        X[i].extend(formatted_row)
    return X, headers

def run(args):
    # optimizemethod="Powell"  #this just does not seem to do very well 
    optimizemethod="Nelder-Mead" # this works best but not perfect,  take the best of 3 calls 
    
    isfolded = args.foldstatus == "isfolded" 
    dofolded = args.foldstatus == "isfolded"  or args.foldstatus == "foldit"
    maxi = args.maxi
    densityof2Ns = args.densityof2Ns
    use_theta_ratio = args.use_theta_ratio # if True use watterson estimator for thetaN
    fix_theta_ratio = args.fix_theta_ratio 
    estimate_pointmass0 = args.estimate_pointmass0
    numtries = args.optimizetimes
    fixmode0 = args.fixmode0
    if args.globalopt != None:
        assert args.globalopt in ('b','s')
    gdensitystart = [1.0,1.0]
    estimatemax2Ns = False
    fixedmax2Ns = args.max2Ns
    if densityof2Ns in ("lognormal","gamma") and fixmode0 == False:
        estimatemax2Ns = args.max2Ns is None
    numparams = 0
    numparams += 0 if fix_theta_ratio else (1 if use_theta_ratio else 2)
    numparams += 2 if densityof2Ns in ("normal","lognormal","gamma","discrete3") else (1 if densityof2Ns=="single2Ns" else 0) 
    numparams += 1 if estimatemax2Ns else 0 
    numparams += 1 if estimate_pointmass0 else 0 
    sfsfilename = args.parent_dir
    # sfsfilename = "./drosophila/foldedsfs_shortintrons_nonsynonymous_pairs.txt"
    #sfsfilename = "./drosophila/foldedsfs_shortintrons_synonymous_pairs.txt"
    # sfsfilename = "./drosophila/foldedsfs_synonymous_nonsynonymous_pairs.txt"
    label = args.outfilelabel #"NMx20_plus_basinhopping_x "
    if fix_theta_ratio:
        label += "_fixthetaratio"
    if estimate_pointmass0:
        label += "_pointmass0"
    if fixmode0:
        label += "_fixmode0"
    if args.densityof2Ns in ("normal","lognormal","gamma","discrete3") :
        if "nonsynonymous" in sfsfilename:
            # outfn = sfsfilename[:-4] + "_" + args.densityof2Ns + "_" +label  + "_estimates.out"

            outfn = op.join(op.split(sfsfilename)[0],"NONSynonymous_" + args.densityof2Ns + "_" +label  + "_estimates.out")
        else:
            outfn = op.join(op.split(sfsfilename)[0],"Synonymous_" + args.densityof2Ns + "_" +label  + "_estimates.out")
    else:
        if "nonsynonymous" in sfsfilename:
            outfn = outfn = op.join(op.split(sfsfilename)[0],"NONSynonymous_single_g_" + label  + "_estimates.out")
        else:
            outfn = outfn = op.join(op.split(sfsfilename)[0],"Synonymous_single_g_" + label  + "_estimates.out")
    outf = open(outfn, "w")
    outf.write("Program run_one_pair_of_SFSs.py results\n========================================\n")
               
    outf.write("Command line: " + args.commandstring + "\n")
    outf.write("Command line arguments:\n")
    for key, value in vars(args).items():
        outf.write("\t{}: {}\n".format(key,value))
    outf.write("\n")#{} ifolded:{} dofolded:{} densityof2Ns:{} use_theta_ratio:{} max2Ns:{} label:{}\n".format(sfsfilename,isfolded,dofolded,densityof2Ns,use_theta_ratio,args.max2Ns, label))
    nc,numsfss,ssfs,fsfss = getsfss(sfsfilename,isfolded=isfolded)
    if dofolded:
        if isfolded==False:
            # this folding assumes nc is even 
            neusfs = [0] + [ssfs[0][j]+ssfs[i][nc -j] for j in range(1,nc //2)] + [ssfs[0][nc //2]]
            selsfs = [0] + [ssfs[1][j]+ssfs[i+1][nc -j] for j in range(1,nc //2)] + [ssfs[1][nc //2]]
        else:
            neusfs = fsfss[0]
            selsfs = fsfss[1]
    else:
        neusfs = ssfs[0]
        selsfs = ssfs[1]
    ratios = [math.inf if  neusfs[j] <= 0.0 else selsfs[j]/neusfs[j] for j in range(len(neusfs))]
    thetaNest = sum(neusfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    thetaSest = sum(selsfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    thetaratioest = thetaNest/thetaSest

    # Initialize the table and headers
    X = []
    headers = []
    X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["DataN","DataS","DataRatio"])
    
    resultlabels =["likelihood"]
    resultformatstrs = ["{}\t{:.3f}"]
    bounds = []
    if use_theta_ratio:
        func = SFS_functions.NegL_SFSRATIO_estimate_thetaratio
        arglist = (nc,dofolded,densityof2Ns,fix_theta_ratio,fixedmax2Ns,estimate_pointmass0,fixmode0,ratios)
        if fix_theta_ratio is None:
            resultlabels += ["thetaratio"]
            resultformatstrs += ["{}\t{:.4f}\t({:.4f} - {:.4f})"]
            bounds.append((thetaratioest/20,thetaratioest*20))
    else:
        resultlabels += ["thetaN","thetaS"]
        resultformatstrs += ["{}\t{:.2f}\t({:.2f} - {:.2f})","{}\t{:.2f}\t({:.2f} - {:.2f})"]
        bounds += [(thetaNest/30,thetaNest*30),(thetaSest/30,thetaSest*30)]
        func = SFS_functions.NegL_SFSRATIO_estimate_thetaS_thetaN
        arglist = (nc,dofolded,densityof2Ns,False,fixedmax2Ns,estimate_pointmass0,fixmode0,ratios)     

    # if densityof2Ns in ("lognormal","gamma"):
    #     bounds += [(0.001,10),(0.001,10)]
    if densityof2Ns == "lognormal":
        bounds += [(-5,10),(0.00001,10)]
    elif densityof2Ns =="gamma":
        bounds += [(0.00001,10),(0.00001,10)]        
    elif densityof2Ns=="normal":
        bounds += [(-20,20),(0.001,10)]
    elif densityof2Ns=="discrete3":
        d3ppos = len(bounds)
        bounds += [(0.0,1.0),(0.0,1.0)]
    else:# otherwise density == "single2Ns"
        bounds += [(-1000,1000)]
    if densityof2Ns in ("lognormal","normal"):
        resultlabels += ["mu","sigma"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})","{}\t{:.5f}\t({:.5f} - {:.5f})"]
    elif densityof2Ns == "gamma":
        resultlabels += ["alpha","beta"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})","{}\t{:.5f}\t({:.5f} - {:.5f})"]
    elif densityof2Ns == "discrete3":
        resultlabels += ["p0","p1"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})","{}\t{:.5f}\t({:.5f} - {:.5f})"]        
    elif densityof2Ns == "single2Ns":
        resultlabels += ["2Ns"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})"]
    else:
        print("density error")
        exit()
    if estimate_pointmass0:
        resultlabels += ["pm0"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})"]
        bounds += [(0.0,1.0)]
    if estimatemax2Ns:
        resultlabels += ["max2Ns"]
        resultformatstrs += ["{}\t{:.5f}\t({:.5f} - {:.5f})"]
        bounds += [(-20.0,20.0)]
    resultlabels += ["expectation","mode"]
    resultformatstrs += ["{}\t{:.3f}","{}\t{:.3f}"]
    paramlabels = resultlabels[1:-2] # result labels includes likelihood in pos 0 and expectation in pos -1,  the rest are parameter labels 
    if numtries > 0:
        outf.write("trial\t" + "\t".join(paramlabels) + "\n")
        resvals = []
        rfunvals = []
        holdbounds = []
    for ii in range(numtries):
        # print("try",ii)
        boundsarray = [
            (random.uniform(bounds[i][0],bounds[i][0] + 0.3*(bounds[i][1]-bounds[i][0])),
                random.uniform(bounds[i][1] - 0.3*(bounds[i][1]-bounds[i][0]),bounds[i][1]) ) for i in range(len(bounds))]
        if densityof2Ns=="discrete3":
            boundsarray[d3ppos] = (0,1)
            boundsarray[d3ppos+1] = (0,1)
        holdbounds.append(boundsarray)
        startarray = [(boundsarray[i][0] + boundsarray[i][1])/2.0 for i in range(len(boundsarray))]
        if densityof2Ns=="discrete3":
            startarray[d3ppos] = 0.333
            startarray[d3ppos+1] = 0.333
        # if fixmode0:
        #     startarray = checkmodestart(densityof2Ns,use_theta_ratio,startarray,boundsarray)
        holdbounds.append(boundsarray)
        result = minimize(func,np.array(startarray),args=arglist,bounds = boundsarray,method=optimizemethod,options={"disp":False,"maxiter":1000*4})                             
        resvals.append(result)
        rfunvals.append(-result.fun)
        outf.write("{}\t{:.5f}\t{}\n".format(ii,-result.fun," ".join(f"{num:.5f}" for num in result.x)))
    if numtries > 0:
        besti = rfunvals.index(max(rfunvals))
        result = resvals[besti]
        confidence_intervals = SFS_functions.generate_confidence_intervals(func,list(result.x), arglist, max(rfunvals), holdbounds[besti])

        #make variables out of parameter labels 
        paramdic = dict(zip(paramlabels,result.x))
        if use_theta_ratio:
            paramdic['thetaN']=thetaNest
            if fix_theta_ratio is not None:
                paramdic["thetaratio"] = fix_theta_ratio
            paramdic['thetaS'] = paramdic["thetaratio"]*thetaNest
        if fixmode0:
            if densityof2Ns=="lognormal":
                fixedmax2Ns = math.exp(paramdic['mu'] - pow(paramdic['sigma'],2))
            if densityof2Ns=="gamma":
                if paramdic['alpha'] < 1:
                    fixedmax2Ns = 0.0
                else:
                    fixedmax2Ns =  (paramdic['alpha'] - 1)*paramdic['beta']
        #get expectation
        if densityof2Ns == "lognormal":
            _,expectation,mode = SFS_functions.getdensityadjust(densityof2Ns,(paramdic['max2Ns'] if estimatemax2Ns else fixedmax2Ns),(paramdic['mu'],paramdic['sigma']))
        elif densityof2Ns == "gamma" :
            _,expectation,mode = SFS_functions.getdensityadjust(densityof2Ns,(paramdic['max2Ns'] if estimatemax2Ns else fixedmax2Ns),(paramdic['alpha'],paramdic['beta']))
        elif densityof2Ns == "discrete3":
            expectation = -(11/2)* (-1 + 92*paramdic['p0'] + paramdic['p1'])
            mode = np.nan
        elif densityof2Ns == "normal": 
            expectation = paramdic['mu']
            mode = paramdic['mu']
        else:
            expectation = mode = np.nan
        if estimate_pointmass0:
            expectation *= (1-paramdic['pm0'])
            pm0tempval = paramdic['pm0']
        else:
            pm0tempval = None
        paramwritestr=[resultformatstrs[0].format(resultlabels[0],rfunvals[besti])]
        paramwritestr += [resultformatstrs[i+1].format(val,result.x[i],confidence_intervals[i][0],confidence_intervals[i][1]) for i,val in enumerate(resultlabels[1:-2])]
        paramwritestr += [resultformatstrs[-2].format(resultlabels[-2],expectation)]
        paramwritestr += [resultformatstrs[-1].format(resultlabels[-1],mode)]
        outf.write("\nmiminize best : result.message {}\n".format(result.message))
        outf.write("\n".join(paramwritestr)+"\n")
        if fixmode0:
            outf.write("\nmax2Ns ({:.4f}) determined as a function of density parameters in order to fix density mode at 0 (runtime flag -o)\n".format(fixedmax2Ns))
        if fixedmax2Ns is not None:
            outf.write("\nSet value of max2Ns: {:.4f})\n".format(fixedmax2Ns))
                
        outf.close()
        
        tempthetaratio = paramdic['thetaratio'] if use_theta_ratio else None
        if densityof2Ns=="lognormal":
            if estimatemax2Ns:   
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]),pm0tempval, True, tempthetaratio)
            else:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],fixedmax2Ns,nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]), pm0tempval, True, tempthetaratio)
        elif densityof2Ns=='gamma':
            if estimatemax2Ns:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,dofolded,densityof2Ns,(paramdic["alpha"],paramdic["beta"]), pm0tempval, True, tempthetaratio)
            else:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],fixedmax2Ns,nc ,None,dofolded,densityof2Ns,(paramdic["alpha"],paramdic["beta"]), pm0tempval, True, tempthetaratio)
        elif densityof2Ns=="normal":
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]), pm0tempval, True, tempthetaratio)
        elif densityof2Ns == "discrete3":
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["p0"],paramdic["p1"]), pm0tempval, True, tempthetaratio)                
        else:  #densityof2Ns =="single2Ns"
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["2Ns"],), pm0tempval, True, tempthetaratio)

        X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["NM{}xN".format(numtries),"NM{}xS".format(numtries),"NM{}xRatio".format(numtries)])

    #basinhopping or shgo
    if args.globalopt != None:
        boundsarray = bounds
        startarray = [(boundsarray[i][0] + boundsarray[i][1])/2.0 for i in range(len(boundsarray))]
        if args.globalopt == 'b':
            result = basinhopping(func,np.array(startarray),T=10.0,
                                    minimizer_kwargs={"method":optimizemethod,"bounds":boundsarray,"args":arglist})
        elif args.globalopt == 's':
            result = shgo(func,boundsarray,args=arglist,minimizer_kwargs={"method":optimizemethod})       
        # print("done",result.x)
        likelihood = -result.fun
        confidence_intervals = SFS_functions.generate_confidence_intervals(func,list(result.x), arglist, likelihood,boundsarray)
        paramdic = dict(zip(paramlabels,result.x))
        if use_theta_ratio:
            paramdic['thetaN']=thetaNest
            paramdic['thetaS'] = paramdic["thetaratio"]*thetaNest
        if fixmode0:
            if densityof2Ns=="lognormal":
                fixedmax2Ns = math.exp(paramdic['mu'] - pow(paramdic['sigma'],2))
            if densityof2Ns=="gamma":
                if paramdic['alpha'] < 1:
                    fixedmax2Ns = 0.0
                else:
                    fixedmax2Ns =  (paramdic['alpha'] - 1)*paramdic['beta']            
        #get expectation
        if densityof2Ns == "lognormal":
            _,expectation,mode = SFS_functions.getdensityadjust(densityof2Ns,(paramdic['max2Ns'] if estimatemax2Ns else fixedmax2Ns),(paramdic['mu'],paramdic['sigma']))
        elif densityof2Ns == "gamma" :
            _,expectation,mode = SFS_functions.getdensityadjust(densityof2Ns,(paramdic['max2Ns'] if estimatemax2Ns else fixedmax2Ns),(paramdic['alpha'],paramdic['beta']))
        elif densityof2Ns == "normal":
            expectation = paramdic['mu']
            mode = paramdic['mu']
        else:
            expectation = mode = np.nan
        if estimate_pointmass0:
            expectation *= (1-paramdic['pm0'])
        paramwritestr=[resultformatstrs[0].format(resultlabels[0],likelihood)]
        paramwritestr += [resultformatstrs[i+1].format(val,result.x[i],confidence_intervals[i][0],confidence_intervals[i][1]) for i,val in enumerate(resultlabels[1:-2])]
        paramwritestr += [resultformatstrs[-2].format(resultlabels[-2],expectation)]        
        paramwritestr += [resultformatstrs[-1].format(resultlabels[-1],mode)]        
        # print(result)   
        # print(result.message)
        outf = open(outfn, "a")
        try:
            if args.globalopt == 'b':
                outf.write("\nbasinhopping best : result.message {}\n".format(result.message))
            elif args.globalopt == 's':
                outf.write("\nshgo best : result.message {}\n".format(result.message))
        except:
            pass
        outf.write("\n".join(paramwritestr)+"\n") 
        if estimatemax2Ns == False and densityof2Ns != "single2Ns" and fixedmax2Ns is not None:
            if fixmode0:
                outf.write("\nmax2Ns ({:.4f}) determined as a function of density parameters in order to fix density mode at 0 (runtime flag -o)\n".format(fixedmax2Ns))
            else:
                outf.write("\nSet value of max2Ns: {:.4f}\n".format(fixedmax2Ns))
        tempthetaratio = paramdic['thetaratio'] if use_theta_ratio else None
        if densityof2Ns=="lognormal":
            if estimatemax2Ns:   
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]), True,tempthetaratio)
            else:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],fixedmax2Ns,nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]), True,tempthetaratio)
        elif densityof2Ns=='gamma':
            if estimatemax2Ns:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,dofolded,densityof2Ns,(paramdic["alpha"],paramdic["beta"]), True,tempthetaratio)
            else:
                neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],fixedmax2Ns,nc ,None,dofolded,densityof2Ns,(paramdic["alpha"],paramdic["beta"]), True,tempthetaratio)
        elif densityof2Ns=="normal":
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["mu"],paramdic["sigma"]), pm0tempval, True, tempthetaratio)
        elif densityof2Ns == "discrete3":
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["p0"],paramdic["p1"]), pm0tempval, True, tempthetaratio)
        else:  #densityof2Ns == "single2Ns"
            neusfs,selsfs,ratios = SFS_functions.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,dofolded,densityof2Ns,(paramdic["2Ns"],), pm0tempval, True, tempthetaratio)                
        if args.globalopt == 'b':
            X, headers = update_table(X, headers,[neusfs,selsfs,ratios],["BsnHp_N","BsnHp_S","BsnHp_Ratio"])
        elif args.globalopt == 's':
            X, headers = update_table(X, headers,[neusfs,selsfs,ratios],["shgo_N","shgo_S","shgo_Ratio"])
        outf.close()
    outf = open(outfn, "a")
    outf.write("\ncompare data and estimates\n--------------------------\n") 
    # Write the headers
    outf.write("i\t" + "\t".join(headers) + "\n")
    
    # Write the data
    for i,row in enumerate(X):
        outf.write("{}\t".format(i) +"\t".join(row) + "\n")            

    endtime = time.time()
    total_seconds = endtime-starttime
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    outf.write(f"\nTime taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n")            
    outf.close()
    print("done ",outfn)

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="parent_dir",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-c",dest="fix_theta_ratio",default=None,type=float,help="set the fixed value of thetaS/thetaN")
    parser.add_argument("-d",dest="densityof2Ns",default = "single2Ns",type=str,help="gamma or lognormal or normal, only if simulating a distribution of Ns, else single values of 2Ns are used")
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-g",dest="globalopt",default=None,help="b for use basinhopping (best but slow),  s for use shgo algorithm, default is neither") 
    parser.add_argument("-i",dest="optimizetimes",type=int,default=0,help="run the minimize optimizer # times")
    parser.add_argument("-l",dest="outfilelabel",default = "", type=str, help="string for labelling resutls file")    
    parser.add_argument("-m",dest="max2Ns",default=None,type=float,help="optional setting for the maximum 2Ns, use only with -d lognormal or -d gamma ")
    parser.add_argument("-M",dest="maxi",default=None,type=int,help="optional setting for the maximum bin index to include in the calculations")
    parser.add_argument("-o",dest="fixmode0",action="store_true",default=False,help="fix the mode of 2Ns density at 0, only works for lognormal and gamma") 
    parser.add_argument("-w",dest="use_theta_ratio",action="store_true",default=False,help="do not estimate both thetas, just the ratio") 
    parser.add_argument("-z",dest="estimate_pointmass0",action="store_true",default=False,help="include a proportion of the mass at zero in the density model, requires normal, lognormal or gamma") 
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    if args.densityof2Ns not in ("lognormal","gamma"):
        if (args.maxi is not None or args.max2Ns is not None):
            parser.error('cannot use -M or -m with -d normal or fixed 2Ns value')
        if (args.fixmode0):
            parser.error('cannot use -o, fixmode0,  with normal density or fixed 2Ns value')
    if args.densityof2Ns=="discrete3" and (args.fixmode0 or args.estimate_pointmass0):
        parser.error('cannot use -o or -z with discrete3 model')
    if args.fixmode0:
        if args.max2Ns:
            parser.error(' cannot use -o, fixmode0,  and also fix max2Ns (-m)')
        if args.estimate_pointmass0:
            parser.error(' cannot have point mass at 0 and fixed mode at 0')
    if args.fix_theta_ratio and args.use_theta_ratio == False:
        parser.error(' cannot use -c fix_theta_ratio without -w use_theta_ratio ')
    return args


if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    run(args)