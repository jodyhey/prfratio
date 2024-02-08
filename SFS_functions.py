"""
    poisson random field SFS work 
    a module of various functions
    see notebook SFS_ratio_modelling.nb

    sfs lists:
        all sfs lists begin with 0 in position 0 
        there is no position for a count where all chromosomes have the allele (i.e. fixed in the sample)


    counting with k chromosomes
        with unfolded, there are k - 1 values  so n_unf = k-1
            this means an unfolded list has length 1 + k - 1 == k 
        with folded it is more complicated, 
            if k is even,  a count of k//2  has no folding partner. e.g. if k is 4  then when folding the bins for counts 1, 3 are summed
                but the bin for 2 is not added to anything.  
                so n_f  has k//2 values  i.e. n_f = k//2  in a list that has length 1 + n_f
            if k is odd, a count of k//2 does have a folding partner, e.g. if k is 5 then bins 1,3 are summed, as are 2,4 
                so n_f has k//2 values,  i.e. n_f = k//2   in a list that has length 1 + n_f
            so the folded value of n_f for even k is the same as for odd count of k + 1 
    nc : # of chromosomes
    n_unf : nc - 1
    n_f : nc // 2 

        
"""
import numpy as np
import  mpmath 
import math
from scipy.optimize import minimize, brentq
import scipy
from scipy import integrate
from scipy.optimize import golden
from scipy.special import erfc, gamma, gammainc,gammaincc
from scipy.stats import chi2


# import warnings
# # warnings.simplefilter('error')
# # Configure warnings to treat RuntimeWarning as an exception
# warnings.simplefilter("error", RuntimeWarning)
# def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
#     print("XXX",message,"/n", category,"\n",filename,"\n",lineno)
#     raise category(message)
# # Register the custom warning filter
# warnings.showwarning = custom_warning_handler

#function naming: primary elements if main likelihood function names 
#   NegL if negative of likelihood
#   SFS  or SFSRATIO 
#    estimate_thetaratio  or estimate_thetaS_thetaN 


#for the Jacobian when using BFGS optimization in SFS_estimator_bias_variance.py
# from here  https://stackoverflow.com/questions/41137092/jacobian-and-hessian-inputs-in-scipy-optimize-minimize
# ended up mostly, not alwyas, crashing with nan errors 
# from numdifftools import Jacobian
# def NegL_SFSRATIO_Theta_Nsdensity_der(x, max2Ns,nc ,dofolded,densityof2Ns,zvals):
#     return Jacobian(lambda x: NegL_SFSRATIO_Theta_Nsdensity(x,max2Ns,nc ,dofolded,densityof2Ns,zvals))(x).ravel()
   

#constants
# as of 8/23/2023 Kuethe method does not work as well as that of  Díaz-Francés, E. and F. J. Rubio
# if False then use the ratio probability function of Díaz-Francés, E. and F. J. Rubio, which works better 
SSFconstant_dokuethe = False 
sqrt2 =pow(2,1/2)
sqrt_2_pi = np.sqrt(2 * np.pi)
sqrt_pi_div_2 = np.sqrt(np.pi/2)

## an array of 2Ns values spanning a useful range,  used for numerically integrating over the density of 2Ns
norm_g_xvals = np.concatenate([np.array([-100,-50,-40,-30,-20]),np.linspace(-19,-4,30),np.linspace(-4,4,50),np.linspace(4,19,30),np.array([20,30,40,50,100])])
# for lognormal and gamma distributions, from 0 to 10000, gets shifted by  subtracting from the maximum of 2Ns 
base_g_xvals = np.concatenate([np.array([1000,900,800,700,600,500,400,300,200,175,150,125,100,90,80,70,60,50,40,30,20]),np.linspace(19,2.1,50),np.linspace(2,1e-5,40)])
discrete3_xvals = np.concatenate([np.linspace(-1000,-2,20),np.linspace(-1,1,20),np.linspace(1.1,10,20)])

last_max2Ns = 1.0 # some routines update max2Ns,  when it changes g_xvals must be updated with a call to check_g_xvals
# g_xvals is the working array,  from -10000+max2Ns to max2Ns, gets reset whenever max2Ns changes 
g_xvals = last_max2Ns - base_g_xvals

thetaNspace = np.logspace(2,4,num=20)

def reset_g_xvals(max2Ns):
    """
        reset g_xvals for a new max value
    """
    global g_xvals
    g_xvals = max2Ns - base_g_xvals

def check_g_xvals(max2Ns):
    """
        if max2Ns has changed  do a reset of g_vals
    """
    global last_max2Ns
    if max2Ns != last_max2Ns:
        last_max2Ns = max2Ns
        reset_g_xvals(max2Ns)

def coth(x):
    """
        save a bit of time for large values of x 
    """
    if abs(x) > 15: # save a bit of time 
        return -1.0 if x < 0 else 1.0
    else:
        return np.cosh(x)/np.sinh(x) 

def logprobratio(alpha,beta,z):
    """
        returns the log of the probability of a ratio z of two normal densities when for each normal density the variance equals the mean 
        is called from other functions,  where alpha, beta and the ratio z have been calculated for a particular frequency bin

        two versions of this function
       
        Díaz-Francés, E. and F. J. Rubio paper expression (1)

        Kuethe DO, Caprihan A, Gach HM, Lowe IJ, Fukushima E. 2000. Imaging obstructed ventilation with NMR using inert fluorinated gases. Journal of applied physiology 88:2279-2286.
        the gamma in the paper goes away because it is equal to 1/alpha^(1/2) when we assume the normal distributions have mean equal to variance 

        The functions are similar, but the Díaz-Francés, E. and F. J. Rubio  works much better overall,  e.g. LLRtest check and ROC curves are much better 
        However the Díaz-Francés and Rubio function gives -inf often enough to cause problems for the optimizer,  so we set final probability p = max(p,1e-50)

    """
    
    if SSFconstant_dokuethe: # not in use
        try:
            alpha2 = alpha*alpha
            alphaR = math.sqrt(alpha)
            z2 = z*z 
            beta2 = beta*beta
            z2b1term = 1+z2/alpha
            xtemp = -(alpha+1)/(2*beta2)
            if xtemp < -709:
                temp1 = 0.0
            else:
                temp1num = math.exp(xtemp)
                temp1denom = math.pi*alphaR*z2b1term
                temp1 = temp1num/temp1denom
            xtemp = -pow(z-alpha,2)/(2*alpha2*beta2*z2b1term)
            if xtemp < -709: # exp of this is zero and so the entire 2nd term of the probability will be zero 
                p = temp1
            else:
                temp2num1 = (1+z)*math.exp(xtemp)
                temp2num2 = math.erf((z+1)/(sqrt2*beta*math.sqrt(z2b1term)))
                temp2denom = sqrt_2_pi * alphaR*beta*pow(z2b1term,1.5)
                p = temp1 + (temp2num1*temp2num2)/temp2denom
        except RuntimeWarning as rw:
            print(f"Caught a RuntimeWarning: {rw}")
        except Exception as e:
            print(f"Caught an exception: {e}")        
        if p > 0:
            return math.log(p)
        else:
            return -math.inf
    else: # Díaz-Francés, E. and F. J. Rubio 
        # some tricky floating point issues,  have to use mpmath 
        try:
            delta = beta
            beta = alpha
            z2 = z*z
            delta2 = delta*delta
            z1 = 1+z
            z2b1 = 1+z2/beta
            z2boverb = (z2+beta)/beta
            betasqroot = math.sqrt(beta)
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            try:
                ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            except:
                ratiotemp2 = mpmath.fdiv(mpmath.fneg(mpmath.power(-z-beta,2)),(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = mpmath.fadd(temp1,temp2)
            # print("{:.1f} {}".format(z,float(p)))
            logp = float(mpmath.log(p))
            return logp 
        except Exception as e:
            print("Caught an exception in logprobratio: {}  p {}".format(e,p))  
            exit()

def intdeltalogprobratio(alpha,z,deltavals):
    """
        integrates over the delta term in the probability of a ratio of two normal distributions
    """
    def trapezoidal_integration(x_values, y_values):
        """Performs trapezoidal integration over lists of floats and function values."""
        integral = 0
        for i in range(1, len(x_values)):
            dx = x_values[i] - x_values[i - 1]
            y_avg = (y_values[i] + y_values[i - 1]) / 2
            integral += dx * y_avg
        return math.log(integral)
    
    def rprobdelta(z,beta,delta):
        delta2 = delta*delta
        try:
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            try:
                ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            except:
                ratiotemp2 = mpmath.fdiv(mpmath.fneg(mpmath.power(-z-beta,2)),(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = float(mpmath.fadd(temp1,temp2))
            # print("{:.1f} {}".format(z,float(p)))
            return p
            logp = float(mpmath.log(p))
            return logp 
        except Exception as e:
            print(f"Caught an exception in rprobdelta: {e}")  
            # return -10000    

    beta = alpha
    z2 = z*z
    z1 = 1+z
    z2b1 = 1+z2/beta
    z2boverb = (z2+beta)/beta
    betasqroot = math.sqrt(beta)
    rprob_density_values = np.array([rprobdelta(z,beta,delta) for delta in deltavals])
    rprob=np.trapz(np.flip(rprob_density_values),np.flip(deltavals)) # have to flip as trapz expects increasing values 
    logrprob = math.log(rprob)
    return logrprob

def ztimesprobratio(z,alpha,beta,doneg):
    """
        called by ratio_expectation()
        for getting the expected ratio
        doneg is True when called by golden()
        False when doing the integration
    """

    # Díaz-Francés, E. and F. J. Rubio 
    try:
        delta = beta
        beta = alpha
        z2 = z*z
        delta2 = delta*delta
        z1 = 1+z
        z2b1 = 1+z2/beta
        z2boverb = (z2+beta)/beta
        betasqroot = math.sqrt(beta)
        ratiotemp = -(1+beta)/(2*delta2)
        temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
        ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
        temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
        temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
        temp2 = mpmath.fdiv(temp2num,temp2denom )
        p = float(mpmath.fadd(temp1,temp2))
        # print("{:.1f} {}".format(z,float(p)))
        # logp = float(mpmath.log(p))  ?
    except Exception as e:
        print(f"Caught an exception in ztimesprobratio: {e}")  
    if p < 0.0:
        return 0.0
    if doneg:
        return -p*z
    return  p*z


def ratio_expectation(p,i,max2Ns,nc,dofolded,densityof2Ns):
    """
    get the expected ratio for bin i given a set of parameter values 
    """
    
    def ztimesprobratio(z,alpha,beta,doneg):
        """
            called by ratio_expectation()
            for getting the expected ratio
            doneg is True when called by golden()
            False when doing the integration
        """

        # Díaz-Francés, E. and F. J. Rubio 
        try:
            delta = beta
            beta = alpha
            z2 = z*z
            delta2 = delta*delta
            z1 = 1+z
            z2b1 = 1+z2/beta
            z2boverb = (z2+beta)/beta
            betasqroot = math.sqrt(beta)
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = float(mpmath.fadd(temp1,temp2))
            # print("{:.1f} {}".format(z,float(p)))
            # logp = float(mpmath.log(p))  ?
        except Exception as e:
            print(f"Caught an exception in ztimesprobratio: {e}")  
        if p < 0.0:
            return 0.0
        if doneg:
            return -p*z
        return  p*z    

    foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not, assumes nc is even, in which case the last bin is not folded 
    thetaN = p[0]
    thetaS = p[1]
    g = (p[2],p[3])
    tempxvals = norm_g_xvals if densityof2Ns=="normal" else g_xvals
    densityadjust,expectation,mode =  getdensityadjust(densityof2Ns,max2Ns,g)
    density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,g[0],g[1],max2Ns,densityof2Ns,foldxterm) for x in tempxvals])
    ux=float(thetaS*np.trapz(density_values,tempxvals))
    uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
    alpha = ux/uy
    sigmay = math.sqrt(uy)
    beta = 1/sigmay
    peak = golden(ztimesprobratio,args=(alpha,beta,True), brack = (0,1000)) # use ratio interval of 0 1000 to start 
    x = integrate.quad(ztimesprobratio,-10,peak*10,args=(alpha,beta, False)) # use interval of 0 to 10* the peak location 
    return x[0]

    

def prf_selection_weight(nc ,i,g,dofolded):
    """
        Poisson random field selection weight for g=2Ns for bin i  (folded or unfolded)
        this is the function you get when you integrate the product of two terms:
             (1) WF term for selection    (1 - E^(-2 2 N s(1 - q)))/((1 - E^(-2 2 N s)) q(1 - q))  
             (2) bionomial sampling formula for i copies,  given allele frequency q 
        over the range of allele frequencies 

        12/18/2023 the hyp1f1 function can easily fail.  I spent a while 
        looking at it and using mpmath to see what values were failing. 
        From what I could tell it is always when the function is trying to return a positive value near 0
        I could use mpmath,  but it does not seem to matter,  as they values are effectively 0
        So for dofolded, so I just set them to 0
        I use mpmath when dofolded = False,  as there is only one hpy1f1 call
    """
    if g==0:
        if dofolded:
            us = nc/(i*(nc -i))
        else:
            us = 1/i
        return us
    tempc = coth(g)
    if tempc==1:
        if dofolded:
            us = 2*(nc /(i*(nc -i)))
        else:
            us = (nc /(i*(nc -i)))
    if dofolded:
        try:
            temph1 = scipy.special.hyp1f1(i,nc ,2*g)
        except: # Exception as e:
            # print(f"Caught an exception: {e}") 
            temph1 = float( mpmath.hyp1f1(i,nc,2*g))
        try: 
            temph2 = scipy.special.hyp1f1(nc - i,nc ,2*g)
        except: # Exception as e:
            # print(f"Caught an exception: {e}") 
            temph2 = float(mpmath.hyp1f1(nc-i,nc,2*g))
        temph = temph1+temph2
        us = (nc /(2*i*(nc -i)))*(2 +2*tempc - (tempc-1)*temph)
    else:
        try:
            temph = scipy.special.hyp1f1(i,nc ,2*g)
        except:
            temph = float(mpmath.hyp1f1(i,nc ,2*g))
        us = (nc /(2*i*(nc -i)))*(1 + tempc - (tempc-1)* temph)       

    return us

def getdensityadjust(densityof2Ns,max2Ns,g):
    """
        integrations are only down over a finite range
        need to get the total area under that range in order to rescale probability
        return an adjustment scalar
        useful to get the expectation over the range as well 
    """
    def uga(a,b):
        """
        scipy gammainc() is scaled lower gamma, gammaincc() is scaled upper gamma
        returned unscaled upper gamma

        """ 
        return gamma(a)*gammaincc(a,b)
    
    def prfdensity(g,params,val=None):
        if densityof2Ns=="lognormal":   
            mean = params[0]
            std_dev = params[1]
            x = float(max2Ns-g)
            p = (1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))
            if p==0.0:
                p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))
            if val == None:
                return p
            else:
                return val*p
        elif densityof2Ns=="gamma":
            alpha = params[0]
            beta = params[1]
            x = float(max2Ns-g)
            p = ((x**(alpha-1))*np.exp(-x * beta))/((beta**alpha)*math.gamma(alpha))
            if val == None:
                return p
            else:
                return val*p
            
    if densityof2Ns in ("lognormal","gamma"):
        # for x in g_xvals:
        check_g_xvals(max2Ns)

        density_values = np.array([prfdensity(x,g) for x in g_xvals])
        densityadjust = float(np.trapz(density_values,g_xvals))
        expectationterms = np.array([prfdensity(x,g,val=x/densityadjust) for x in g_xvals])
        expectation = float(np.trapz(expectationterms,g_xvals))
        if densityof2Ns=="lognormal":
            mean = g[0]
            std_dev = g[1]
            mode = max2Ns - math.exp(mean-pow(std_dev,2))
        if densityof2Ns=="gamma":
            alpha = g[0]
            beta = g[1]
            if alpha < 1.0:
                mode = max2Ns
            else:
                mode = max2Ns - (alpha-1.0)*beta
    elif densityof2Ns == "normal":
        expectation = g[0]
        densityadjust = 1.0
        mode = expectation
    return densityadjust,expectation,mode

 
def prfdensityfunction(g,densityadjust,nc ,i,arg1,arg2,max2Ns,densityof2Ns,foldxterm):
    """
    returns the product of poisson random field weight for a given level of selection (g) and a probability density for g 
    used for integrating over g 
    if foldxterm is true,  then it is a folded distribution AND two bins are being summed
    """
    us = prf_selection_weight(nc ,i,g,foldxterm)
    if densityof2Ns=="lognormal":   
        mean = arg1
        std_dev = arg2
        x = float(max2Ns-g)
        p = ((1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2)))/densityadjust
        if p==0.0:
           p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))/densityadjust
    elif densityof2Ns=="gamma":
        alpha = arg1
        beta = arg2
        x = float(max2Ns-g)
        p = (((x**(alpha-1))*np.exp(-x * beta))/((beta**alpha)*math.gamma(alpha)))/densityadjust
    elif densityof2Ns=="normal": # shouldn't need densityadjust for normal
        mu = arg1
        std_dev= arg2
        p = np.exp((-1/2)* ((g-mu)/std_dev)**2)/(std_dev * math.sqrt(2*math.pi))
    elif densityof2Ns=="discrete3":
        if g < -1:
            p=arg1/999
        elif g<= 1:
            p = arg2/2
        else:
            p = (1-arg1-arg2)/9
     
    if p*us < 0.0 or np.isnan(p):
        return 0.0
    return p*us

def NegL_SFS_Theta_Ns(p,nc,dofolded,counts): 
    """
        for fisher wright poisson random field model,  with with selection or without
        if p is a float,  then the only parameter is theta and there is no selection
        else p is a list (2 elements) with theta and Ns values 
        counts begins with a 0
        returns the negative of the log of the likelihood for a Fisher Wright sample 
    """
    # def L_SFS_Theta_Ns_bin_i(p,i,nc,dofolded,count): 
    def L_SFS_Theta_Ns_bin_i(i,count): 
        if isinstance(p,(float, int)): # p is simply a theta value,  no g  
            theta = p
            if theta <= 0:
                return -math.inf
            un = theta*nc/(i*(nc - i)) if dofolded else theta/i
            temp = -un +  math.log(un)*count - math.lgamma(count+1)
        else:
            theta = p[0]
            if theta <= 0:
                return -math.inf
            g = p[1]
            us = theta * prf_selection_weight(nc,i,g,dofolded)
            temp = -us +  math.log(us)*count - math.lgamma(count+1)
        return temp     
    assert(counts[0]==0)
    sum = 0
    for i in range(1,len(counts)):
        # sum += L_SFS_Theta_Ns_bin_i(p,i,nc,dofolded,counts[i])
        sum += L_SFS_Theta_Ns_bin_i(i,counts[i])
    return -sum 


def NegL_SFS_ThetaS_densityNs(p,max2Ns,nc ,dofolded,densityof2Ns,counts):
    """
        basic PRF likelihood
        returns negative of likelihood for the SFS 
        unknowns:
            thetaS
            terms for 2Ns density
    """
    sum = 0
    thetaS = p[0]
    term1 = p[1]
    term2 = p[2]
    if densityof2Ns != "normal":
        check_g_xvals(max2Ns)
    densityadjust,expectation,mode =  getdensityadjust(densityof2Ns,max2Ns,(term1,term2))
    for i in range(1,len(counts)):
        tempxvals = norm_g_xvals if densityof2Ns=="normal" else g_xvals
        density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,term1,term2,max2Ns,densityof2Ns,dofolded) for x in tempvals])
        us=float(thetaS*np.trapz(density_values,tempxvals))
        sum += -us + math.log(us)*counts[i] - math.lgamma(counts[i]+1)        
    return -sum    
 
def NegL_SFSRATIO_estimate_thetaS_thetaN(p,nc,dofolded,densityof2Ns,onetheta,max2Ns,usepm0,fix_mode_0,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        estimates Theta values,  not their ratio

        densityof2Ns in fix2Ns0,single2Ns,normal,lognormal,gamma,discrete3 
        onetheta in True, False
        max2Ns  is either None,  or a fixed max value 
        usepm0 in True, False
        fix_mode_0 in True, False 


        replaces:
            def NegL_SFSRATIO_thetaS_thetaN_fixedNs(p,nc ,dofolded,zvals,nog,usepm0)
            def NegL_SFSRATIO_thetaS_thetaN_densityNs_max2Ns(p,max2Ns,nc ,maxi,dofolded,densityof2Ns,zvals)
            
        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        thetaN,thetaS 1 if onetheta else 2 
        Ns terms    2 if densityNs is not single2Ns 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if usepm0 else 0 

        handles fix_mode_0 i.e. setting max2Ns so the mode of the distribution is 0 
        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("fix2Ns0","single2Ns"):
            try:
                if z==math.inf or z==0.0:
                    return 0.0
                uy = thetaN*nc /(i*(nc -i)) if dofolded else thetaN/i     
                if g == 0:
                    alpha = thetaS/thetaN
                else:
                    ux = thetaS*prf_selection_weight(nc,i,g,dofolded)
                    if usepm0:
                        ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                    alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)
            except:
                return -math.inf                
        else:
            try:
                if densityof2Ns == "discrete3":
                    density_values = np.array([prfdensityfunction(x,None,nc ,i,g[0],g[1],None,densityof2Ns,foldxterm) for x in discrete3_xvals])
                    ux = float(np.trapz(density_values,discrete3_xvals))
                else:
                    tempxvals = norm_g_xvals if densityof2Ns=="normal" else g_xvals
                    density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,g[0],g[1],max2Ns,densityof2Ns,foldxterm) for x in tempxvals])
                    ux = thetaS*np.trapz(density_values,tempxvals)
                    if usepm0:
                        ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
                alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)   
            except:
                return -math.inf 
            
    assert zvals[0]==0 or zvals[0] == math.inf
    try:
        p = list(p)
    except :
        p = [p]
        pass  # if this function is called by scipy minimize then p is just a scalar 
    if onetheta:
        thetaN = thetaS = p[0]
        unki = 1
    else:
        thetaN = p[0]
        thetaS = p[1]
        unki = 2
    if densityof2Ns == "single2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns == "fix2Ns0":
        g = 0
    else:
        g = (p[unki],p[unki+1])
        if densityof2Ns=="discrete3":
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
        holdki = unki
        unki += 2
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        if fix_mode_0:
            if densityof2Ns=="lognormal":
                max2Ns = math.exp(p[holdki] - pow(p[holdki],2))
            if densityof2Ns=="gamma":
                if p[holdki] < 1:
                    max2Ns = 0.0
                else:
                    max2Ns = (p[holdki] - 1)*p[holdki+1]
        else:
            max2Ns = p[unki]
            unki += 1
    if usepm0:
        pm0 = p[unki]
        unki += 1
    if densityof2Ns not in ("single2Ns","discrete3","fix2Ns0"):
        if densityof2Ns != "normal":
            check_g_xvals(max2Ns)
        densityadjust,expectation,mode =  getdensityadjust(densityof2Ns,max2Ns,g)

    sum = 0
    for i in range(1,len(zvals)):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
            # raise Exception("sum==-math.inf in NegL_SFSRATIO_Theta_Nsdensity_given_thetaN params: " + ' '.join(str(x) for x in p))
    # print(-sum,p)
    return -sum   


def NegL_SFSRATIO_estimate_thetaratio(p,nc,dofolded,densityof2Ns,fix_theta_ratio,max2Ns,usepm0,fix_mode_0,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        first parameter is the ratio of mutation rates
        sidesteps the theta terms by integrating over thetaN in the probability of the ratio (i.e. calls intdeltalogprobratio())

        densityof2Ns in fix2Ns0,single2Ns,normal,lognormal,gamma,discrete3 
        fixthetaratio is either None, or a fixed value for the ratio 
        max2Ns  is either None,  or a fixed max value 
        usepm0 in True, False
        fix_mode_0 in True, False 

        replaces:
            NegL_SFSRATIO_ratio_fixedNs
            NegL_SFSRATIO_ratio_densityNs
            NegL_SFSRATIO_ratio_densityNs_pointmass0

        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        ratio       0 if fix_theta_ratio is not None else 1 
        Ns terms    2 if densityNs is not None else 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if usepm0 else 0 

        handles fix_mode_0 i.e. setting max2Ns so the mode of the distribution is 0 
        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("single2Ns","fix2Ns0"):
            try:
                if z==math.inf or z==0.0:
                    return 0.0
                if g == 0:
                    alpha = thetaratio
                else:
                    sint = prf_selection_weight(nc,i,g,foldxterm)
                    if usepm0:
                        sint = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*sint
                    alpha = thetaratio*sint/(nc /(i*(nc -i)) if foldxterm else 1/i )
                uy = thetaNspace * nc /(i*(nc -i)) if foldxterm else thetaNspace/i    
                deltavals = 1/np.sqrt(uy)
                return intdeltalogprobratio(alpha,z,deltavals)        
            except:
                return -math.inf                
        else:
            try:
                if densityof2Ns == "discrete3":
                    density_values = np.array([prfdensityfunction(x,None,nc ,i,g[0],g[1],None,densityof2Ns,foldxterm) for x in discrete3_xvals])
                    sint = float(np.trapz(density_values,discrete3_xvals))
                else:
                    tempxvals = norm_g_xvals if densityof2Ns=="normal" else g_xvals
                    density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,g[0],g[1],max2Ns,densityof2Ns,foldxterm) for x in tempxvals])
                    sint = float(np.trapz(density_values,tempxvals))
                    if usepm0:
                        sint = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*sint
                alpha = thetaratio*sint/(nc /(i*(nc -i)) if foldxterm else 1/i )
                uy = thetaNspace * nc /(i*(nc -i)) if foldxterm else thetaNspace/i    
                deltavals = 1/np.sqrt(uy)
                return intdeltalogprobratio(alpha,z,deltavals)        
            except:
                return -math.inf 
            
    assert zvals[0]==0 or zvals[0] == math.inf
    p = list(p)
    unki = 0
    if fix_theta_ratio is None:
        thetaratio = p[0]
        unki = 1
    else:
        thetaratio = fix_theta_ratio
    if densityof2Ns == "single2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns  == "fix2Ns0":
        g = 0.0
    else:
        g = (p[unki],p[unki+1])
        if densityof2Ns=="discrete3":
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
        unki += 2
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        if fix_mode_0:
            if densityof2Ns=="lognormal":
                max2Ns = math.exp(p[1] - p[2]*p[2])
            if densityof2Ns=="gamma":
                if p[1] < 1:
                    max2Ns = 0.0
                else:
                    max2Ns = (p[1] - 1)*p[2]
        else:
            max2Ns = p[unki]
            unki += 1
    if usepm0:
        pm0 = p[unki]
        unki += 1
    if densityof2Ns in ("normal","lognormal","gamma"):
        if densityof2Ns != "normal":
            check_g_xvals(max2Ns)
        densityadjust,expectation,mode =  getdensityadjust(densityof2Ns,max2Ns,g)

    sum = 0
    for i in range(1,len(zvals)):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
            # raise Exception("sum==-math.inf in NegL_SFSRATIO_Theta_Nsdensity_given_thetaN params: " + ' '.join(str(x) for x in p))
    # print(-sum,p)
    return -sum   

def simsfs_continuous_gdist(theta,max2Ns,nc ,maxi,densityof2Ns, params,pm0, returnexpected):
    """
    nc  is the # of sampled chromosomes 

    simulate the SFS under selection, assuming a PRF Wright-Fisher model 
    uses a distribution of g (2Ns) values 
    gdist is "lognormal" or "gamma" ,params is two values

    return folded and unfolded    
    """
    sfs = [0]*nc 
    for i in range(1,nc):
        if densityof2Ns=="normal":
            tempxvals = norm_g_xvals 
            densityadjust,expectatio,mode =  getdensityadjust(densityof2Ns,max2Ns,params)
        elif densityof2Ns=="discrete3":
            tempxvals = discrete3_xvals
            density_values = np.array([prfdensityfunction(x,None,nc ,i,params[0],params[1],None,densityof2Ns,False) for x in tempxvals])
        elif densityof2Ns in ("lognormal","gamma"):
            check_g_xvals(max2Ns)
            tempxvals = g_xvals
            densityadjust,expectatio,mode =  getdensityadjust(densityof2Ns,max2Ns,params)
            density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,params[0],params[1],max2Ns,densityof2Ns,False) for x in tempxvals])
        sint = np.trapz(density_values,tempxvals)
        if pm0 is not None:
            sint = pm0/i + (1-pm0)*sint
        sfsexp = theta*sint
        assert sfsexp>= 0
        if returnexpected:
            sfs[i] = sfsexp
        else:
            sfs[i] = np.random.poisson(sfsexp)

    sfsfolded = [0] + [sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [sfs[nc //2]]
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded

def simsfs(theta,g,nc ,maxi, returnexpected):
    """
        nc  is the # of sampled chromosomes 

        simulate the SFS under selection, assuming a PRF Wright-Fisher model 
        uses just a single value of g (2Ns), not a distribution
        if returnexpected,  use expected values, not simulated
        generates,  folded and unfolded for Fisher Wright under Poisson Random Field
        return folded and unfolded 
    """
    if g==0:
        sfsexp = [0]+[theta/i for i in range(1,nc )]
    else:
        sfsexp = [0]
        for i in range(1,nc ):
            u = prf_selection_weight(nc ,i,g,False)
            sfsexp.append(u*theta)    
    if returnexpected:
        sfs = sfsexp
    else:    
        sfs = [np.random.poisson(expected) for expected in sfsexp]
    sfsfolded = [0] + [sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [sfs[nc //2]]
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded


def simsfsratio(thetaN,thetaS,max2Ns,nc ,maxi,dofolded,densityof2Ns,params,pm0, returnexpected, thetaratio):
    """
     nc  is the # of sampled chromosomes 

    simulate the ratio of selected SFS to neutral SFS
    if returnexpected,  use expected values, not simulated
    if gdist is None,  params is just a g value,  else it is a list of distribution parameters
    if a bin of the neutral SFS ends up 0,  the program stops

    if ratio is not none, thetaS = thetaratio*thetaN

    pm0 is point mass 0,  as of 2/4/2024 used only by run_one_pair_of_SFSs.py
    """
    
    nsfs,nsfsfolded = simsfs(thetaN,0,nc ,maxi,returnexpected)
    if densityof2Ns in ("lognormal","gamma"):
        check_g_xvals(max2Ns)
    if thetaratio is not None:
        thetaS = thetaN*thetaratio
    if densityof2Ns == "single2Ns":
        ssfs,ssfsfolded = simsfs(thetaS,params[0],nc ,maxi,returnexpected)
    else:
        ssfs,ssfsfolded = simsfs_continuous_gdist(thetaS,max2Ns,nc ,maxi,densityof2Ns,params,pm0,returnexpected)
    if dofolded:
        ratios = [math.inf if nsfsfolded[j] <= 0.0 else ssfsfolded[j]/nsfsfolded[j] for j in range(len(nsfsfolded))]
        return nsfsfolded,ssfsfolded,ratios
    else:
        ratios = [math.inf if nsfs[j] <= 0.0 else ssfs[j]/nsfs[j] for j in range(len(nsfs))]
        return nsfs,ssfs,ratios

def generate_confidence_intervals(func,p_est, arglist, maxLL,bounds,alpha=0.05):
    """
    Generates 95% confidence intervals for each parameter in p_est.
    Find bounds where the likelihood drops by an ammount given by chi-square distribution, 1df 
    Find confidence interval bounds by searching for values that cross the threshold

    Args:
        p_est: List of estimated parameter values.
        f: Function that returns the log-likelihood for a given parameter set.
        maxLL:  the log likelihood at p_est 
        arglist : a tuple or list with all the rest of the arguments,  after the parameter values, to pass to the f 
        alpha: Significance level for confidence intervals (default: 0.05).

    Returns:
        List of tuples, where each tuple contains (lower bound, upper bound) for a parameter.
    """

    confidence_intervals = []
    chiinterval = chi2.ppf(1 - alpha, df=1)/2  # 0.5 for one-sided interval
    likelihood_threshold = maxLL - chiinterval
    checkLL = -func(p_est,*arglist)
    for i in range(len(p_est)):
        # Create a minimization function with fixed parameters except for the i-th one
        def neg_log_likelihood_fixed(p_i):
            p_fixed = p_est.copy()
            p_fixed[i] = p_i
            return -func(p_fixed, *arglist)  # Minimize negative log-likelihood
        
        # widen the bounds of the original search in case CI falls outside the original bounds 
        lbtemp = bounds[i][0]/2
        if lbtemp  <   p_est[i] and  maxLL - neg_log_likelihood_fixed(lbtemp) > chiinterval:
            lower_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, lbtemp, p_est[i])
        else:
            lower_bound = np.nan # this should still be able to be written to the file
        ubtemp = bounds[i][1]*2
        if ubtemp  >   p_est[i] and  maxLL - neg_log_likelihood_fixed(ubtemp) > chiinterval:
            upper_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, p_est[i], ubtemp)
        else:
            upper_bound = np.nan # this should still be able to be written to the file

        confidence_intervals.append((lower_bound, upper_bound))
    return confidence_intervals

