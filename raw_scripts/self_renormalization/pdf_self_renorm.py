# %%
import lsqfit as lsf
from head import *

############################################
####### self-renormalization of pdf ########
############################################

def pdf_zR(plot=False): # main func to extract zR

    ### plot data of bare matrix element ###
    pdf_d = pdf_data_plot(plot=plot) # bare pdf after interpolation into z_ls # 3-D list: z, x, log(mm, ms)

    ### fit to get gz ###
    #####################
    # gz_ls: [z_ls, gz_ls], f1_ls: [z_ls, f1z_ls], z_x: [z_ls, x_ls], lnm: [ ln(mBare)_ls ], gz_fit_result
    gz_ls, f1_ls, z_x, lnm, gz_fit_result = gz_fit(pdf_d, a_milc_ls, a_rbc_ls, plot=plot)

    ### fit to get m0 ###
    #####################
    m0, mR_pdf = m0_fit(gz_ls, plot=plot) # mR_pdf = exp( g(z) - m0z )

    ### plot pdf self-renormolization for check ###
    zR_ls = pdf_fcn(z_x, gz_fit_result.p, zR=True, m0=m0) # zR = exp( ln(M) - g(z) + m0z )
    mR_plot(lnm, zR_ls, z_x, mR_pdf, plot=plot)

    ### plot f1 for extend ###
    ##########################
    if plot == True:
        f1_plot(f1_ls)

    ### extend f1 so that to extend zR #!#
    ######################################
    posterior_extend = f1_extend(gz_fit_result.p, f1_ls, fit_start=7) # fit_start is the idx, comes from f1 plot
    # posterior_extend = gz_fit_result.p # no extend

    ### construct zR_dic for da renormalization ###
    ###############################################
    zR_dic = gv.BufferDict() 
    for a in a_milc_ls:
        z_x_ = {}
        z_x_['milc'] = [[], []]
        z_x_['rbc'] = [[0.06], [1]]
        for z in z_ls_extend:
            z_x_['milc'][0].append(z)
            z_x_['milc'][1].append(gev_fm / a)

        zR_ls_ = pdf_fcn(z_x_, posterior_extend, zR=True, m0=m0_da, d=d_da) #!# here is for da renormalization

        zR_dic['a='+str(a)] = zR_ls_['milc']

    m_pdf_dic = {}
    for ls in pdf_d:
        a = round(gev_fm / ls[1], 4)
        if 'a='+str(a) not in m_pdf_dic:
            m_pdf_dic['a='+str(a)] = []
        m_pdf_dic['a='+str(a)].append( [ls[0], np.exp(ls[2])] ) #append( [z, gvar(M, err)] )

    for key in m_pdf_dic: # sort by z, z in z_ls
        index = np.argsort(m_pdf_dic[key], axis=0)[:,0]
        m_pdf_dic[key] = np.array(m_pdf_dic[key])[index, 1]
    
    return zR_dic, m_pdf_dic

def txt_data(file_name): # read data from txt file
    data = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            lined = np.array([float(i) for i in line.split()])
            if len(lined) != 0:
                data.append(lined)

    return data

def pdf_data_plot(plot=True): 
    data1 = txt_data('clv_Pion_milc1.txt')
    data2 = txt_data('clv_Pion_rbc1.txt')

    # nz, a, nothing, center value of matrix element, uncertainty of matrix element
    raw_d = np.vstack((np.array(data1), np.array(data2)))

    a_dic = {}

    for ls in raw_d:
        z = ls[0] * ls[1]
        a = ls[1]

        if str(a) not in a_dic:
            a_dic[str(a)] = []
        a_dic[str(a)].append([z, ls[3], ls[4]])
    
    pdf_d = [] # pdf matrix element data after interpolation
    for key in a_dic:
        a = float(key)
        x = gev_fm / a # 1/a (GeV)

        z_in = []
        ym_in = []
        ys_in = []
        for ls in a_dic[key]:
            if ls[0] == 0:
                nz0 = ls[1]

        for ls in a_dic[key]:
            z_in.append(ls[0])
            ym_in.append(np.log( abs(ls[1]/nz0) ))
            ys_in.append(np.sqrt( (ls[2]/ls[1])**2 ))
        mm_ls = interp_1d(z_in, ym_in, z_ls) # mean of matrix after interpolation
        ms_ls = interp_1d(z_in, ys_in, z_ls)

        for i in range(len(z_ls)):
            sdev = np.sqrt( ms_ls[i]**2 + (delta_sys * mu / x)**2 ) # error estimation
            pdf_d.append([z_ls[i], x, gv.gvar(mm_ls[i], sdev)]) # z, x, log(mm, ms)

    if plot == True:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        for z in z_ls:
            x = [] # 1/a
            y = [] # ln(m).mean
            err= [] # ln(m).sdev
            for ls in pdf_d:
                if ls[0] == z:
                    x.append(ls[1])
                    y.append(ls[2].mean)
                    err.append(ls[2].sdev)
            if round(z, 2) in [0.3, 0.6, 0.9, 1.2]:
                ax.errorbar(x, y, yerr=err, label='z='+str(round(z, 2)), **errorb)
            else:
                ax.errorbar(x, y, yerr=err, **errorb)
        ax.tick_params(direction='in', **ls_p)
        ax.set_xlabel('1/a', **fs_p)
        ax.set_ylabel('ln(M)', **fs_p)
        ax.set_xlim(1, 8)
        ax.legend()
        ax.set_title('pdf matrix element after interpolation', **fs_p)
        plt.show()

    return pdf_d

def pdf_fcn(z_x, p, zR=False, m0=None, d=None):
    if d == None:
        d = d_pdf

    z_milc = np.array(z_x['milc'][0])
    x_milc = np.array(z_x['milc'][1])
    z_rbc = np.array(z_x['rbc'][0])
    x_rbc = np.array(z_x['rbc'][1])

    val = {}
    n_milc = len(z_milc) # num of milc data points 
    n_rbc = len(z_rbc) # num of rbc data points

    val['milc'] = []
    val['rbc'] = []

    for idx in range(n_milc):
        zm = z_milc[idx]
        xm = x_milc[idx]

        temp = k * zm * xm / gv.log(lqcd / xm) + p['g'+str(zm)] + p['f1'+str(zm)] / xm + 3 * cf / b0 * gv.log( gv.log( xm/lqcd ) / gv.log(mu / lqcd) ) + gv.log( 1 + d / (gv.log(lqcd/xm)) )

        if zR == True:
            temp = temp - p['g'+str(zm)] + m0*zm 

        val['milc'].append(temp)

    for idx in range(n_rbc):
        zr = z_rbc[idx]
        xr = x_rbc[idx]
        temp = k * zr * xr / gv.log(lqcd / xr) + p['g'+str(zr)] + p['f2'+str(zr)] / xr + 3 * cf / b0 * gv.log( gv.log( xr/lqcd ) / gv.log(mu / lqcd) ) + gv.log( 1 + d / (gv.log(lqcd/xr)) ) 

        if zR == True:
            temp = temp - p['g'+str(zr)] + m0*zr 

        val['rbc'].append(temp)

    if zR == True:
        val['milc'] = np.exp(val['milc'])
        val['rbc'] = np.exp(val['rbc'])

    return val

def gz_fit(pdf_d, a_milc_ls, a_rbc_ls, plot=True): # eliminate discrete effects, cancel f1/f2 * a
    fcn = pdf_fcn

    priors = gv.BufferDict()
    for z in z_ls:
        priors['g'+str(z)] = gv.gvar(0, 20)
        priors['f1'+str(z)] = gv.gvar(0, 5)
        priors['f2'+str(z)] = gv.gvar(0, 5)

    z_x = {}
    lnm = {}
    for key in ['milc', 'rbc']:
        z_x[key] = [[], []]
        lnm[key] = []

    for ls in pdf_d:
        if ls[1] in [gev_fm/a for a in a_milc_ls]:
            z_x['milc'][0].append(ls[0]) # z
            z_x['milc'][1].append(ls[1]) # x
            lnm['milc'].append(ls[2]) # ln(m)
                
        elif ls[1] in [gev_fm/a for a in a_rbc_ls]: 
            z_x['rbc'][0].append(ls[0])
            z_x['rbc'][1].append(ls[1]) 
            lnm['rbc'].append(ls[2])

        else:
            print('>>> Fit input wrong')

    fit_result = lsf.nonlinear_fit(data=(z_x, lnm), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

    print(">>>")
    print(fit_result)

    post1 = np.array([val.mean for val in fcn(z_x, fit_result.p)['milc']])
    err1 = np.array([val.sdev for val in fcn(z_x, fit_result.p)['milc']])

    if plot == True:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        ax.errorbar(np.array(z_x['milc'][1]), post1, yerr=err1, color='r', label='fit results', **errorb)
        ax.errorbar(np.array(z_x['milc'][1]), [val.mean for val in lnm['milc']], yerr=[val.sdev for val in lnm['milc']], color='b', label='data', **errorb)
        ax.tick_params(direction='in', **ls_p)
        ax.set_xlim(1, 7)
        ax.set_xlabel('1/a', **fs_p)
        ax.set_ylabel('ln(M)', **fs_p)
        ax.set_title('data v.s. fit results', **fs_p)
        ax.legend(loc='upper right')
        plt.show()

    gz_ls = [[], []]
    for z in z_ls:
        gz_ls[0].append(round(z, 2))
        gz_ls[1].append(fit_result.p['g'+str(z)])

    f1_ls = [[], []]
    for z in z_ls:
        f1_ls[0].append(round(z, 2))
        f1_ls[1].append(fit_result.p['f1'+str(z)])

    return gz_ls, f1_ls, z_x, lnm, fit_result

def m0_fit(gz_ls, plot=True):
    m0z = gz_ls[1] - np.log( ZMS_pdf(np.array(gz_ls[0])) )

    def fcn(x, p):
        val = np.log(ZMS_pdf(np.array(x))) + p['m0']*np.array(x) + p['b']
        return val
    
    print(">>> z_ls")
    print(gz_ls[0][:3])
    
    print('>>> log(MS-bar)')
    print(np.log(ZMS_pdf(np.array(gz_ls[0][:3]))))

    print('>>> gz')
    print(gz_ls[1][:3])

    priors = gv.BufferDict()
    priors['m0'] = gv.gvar(0, 20)
    priors['b'] = gv.gvar(0, 100)

    fit_result = lsf.nonlinear_fit(data=(gz_ls[0][:3], gz_ls[1][:3]), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')
    print(fit_result)

    y1 = (fit_result.p['m0'].mean + fit_result.p['m0'].sdev) * np.arange(0, 0.3, 0.01)
    y2 = (fit_result.p['m0'].mean - fit_result.p['m0'].sdev) * np.arange(0, 0.3, 0.01)

    if plot == True:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        ax.fill_between(np.arange(0, 0.3, 0.01), y1, y2, alpha=0.5, color='r', label='fit results')
        ax.errorbar(gz_ls[0][:4], [val.mean for val in m0z][:4], yerr=[val.sdev for val in m0z][:4], label='data', **errorb)
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([0, 0.4])
        ax.legend(loc='upper right')
        ax.set_title('fit m0 with first three points')
        plt.show()

    zms = ZMS_pdf(np.array(z_ls))
    ## mR_pdf = exp( g(z) - m0z )
    mR_pdf = np.exp( gz_ls[1] - fit_result.p['m0'] * np.array(gz_ls[0]) )

    if plot == True:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        ax.plot(z_ls, zms, 'k-', label='ZMS-bar')
        ax.errorbar(z_ls, [val.mean for val in mR_pdf], yerr=[val.sdev for val in mR_pdf], label='renorm pdf', **errorb)
        ax.errorbar(z_ls, [val.mean for val in mR_pdf / zms], yerr=[val.sdev for val in mR_pdf / zms], color='r', label='ratio', **errorb)
        plt.legend(loc='upper right')
        plt.show()

    print(">>>")
    print(mR_pdf / zms)
    print('fit result of m0: '+str(fit_result.p['m0']))

    return fit_result.p['m0'], mR_pdf

def f1_plot(f1_ls):
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(f1_ls[0], [val.mean for val in f1_ls[1]], yerr=[val.sdev for val in f1_ls[1]], fmt='o', **errorb)
    ax.set_xlabel('z', **fs_p)
    ax.set_ylabel('f1', **fs_p)
    ax.tick_params(direction='in', **ls_p)
    plt.show()

def f1_extend(posterior, f1_ls, fit_start):
    def fcn(x, p):
        return p['a2']*(x**2) + p['a1']*x + p['a0']
    
    priors = gv.BufferDict()
    priors['a0'] = gv.gvar(0, 5)
    priors['a1'] = gv.gvar(0, 5)
    priors['a2'] = gv.gvar(0, 5)

    fit_result = lsf.nonlinear_fit(data=(np.array(f1_ls[0][fit_start:]), f1_ls[1][fit_start:]), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

    for idx in range(len(z_ls), len(z_ls_extend)):
        posterior['f1'+str(z_ls_extend[idx])] = fcn(z_ls_extend[idx], fit_result.p)
        posterior['g'+str(z_ls_extend[idx])] = 0 # will be cancelled in zR

    return posterior

def mR_plot(lnm, zR_ls, z_x, mR_pdf, plot=True): # first 3 variables are dict
    ## mR_ls = mBare / zR_ls
    mR_ls = []
    z_x_ls = [[], []]
    for idx in range(len(lnm['milc'])): # length = l(z) * l(x)
        mR_ls.append( np.exp(lnm['milc'][idx]) / zR_ls['milc'][idx] )
        z_x_ls[0].append( z_x['milc'][0][idx] ) # z
        z_x_ls[1].append( z_x['milc'][1][idx] ) # x

    for idx in range(len(lnm['rbc'])): # length = l(z) * l(x)
        mR_ls.append( np.exp(lnm['rbc'][idx]) / zR_ls['rbc'][idx] )
        z_x_ls[0].append( z_x['rbc'][0][idx] ) # z
        z_x_ls[1].append( z_x['rbc'][1][idx] ) # x

    mR_a_dic = {}
    for idx in range(len(mR_ls)):
        a = round(gev_fm/z_x_ls[1][idx], 4)
        if str(a) not in mR_a_dic:
            mR_a_dic[str(a)] = [[], []]
        mR_a_dic[str(a)][0].append(z_x_ls[0][idx])
        mR_a_dic[str(a)][1].append(mR_ls[idx])

    if plot == True:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        for key in mR_a_dic:
            ax.errorbar(mR_a_dic[key][0], [val.mean for val in mR_a_dic[key][1]], yerr=[val.sdev for val in mR_a_dic[key][1]], label='a='+key, **errorb)
        ax.errorbar(z_ls, [val.mean for val in mR_pdf], yerr=[val.sdev for val in mR_pdf], color='r', label=r'$\exp( g(z) - m_0 z)$', **errorl)
        ax.set_title('PDF_self_renorm', **fs_p)
        ax.set_xlim([0, 1.4])
        ax.legend()
        plt.show()

# %%
if __name__ == "__main__":
    zR_dic, m_pdf_dic = pdf_zR(plot=True)
    print(zR_dic)

# %%
