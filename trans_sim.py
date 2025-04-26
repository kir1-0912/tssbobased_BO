import os
import re
import hashlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from Balun_X_Example_Single_Gen import Balun_X_Gen
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import threading
import math


def exec_emx( fstart=1e8, fstop=6e9, fstep=1e8, CellName="BalunX") :
    if not os.path.exists("./emx_s4p"):
        os.makedirs("./emx_s4p")
    emx_bin_path=r"/home/imsic/zhangwx/EDA/cadence/installs/INTEGRAND60/bin/emx"
    emx_proc_path=r"/home/imsic/zhangwx/EDA/cadence/installs/INTEGRAND60/virtuoso_ui/emxinterface/processes/RC_IRCX_CRN28HPC+_1P9M+UT-ALRDL_6X1Z1U_typical.proc"
    # s4p_file_name=CellName.replace(".","_")
    emx_cmd = f"source /home/imsic/zhangwx/EDA/env_config/env_spectre191_alpsnew_IC618new && {emx_bin_path} {os.getcwd()}/GDS/{CellName}.gds {CellName} {emx_proc_path} -e 1 -t 1 -v 0.5 --3d=* -p P000=P1 -p P001=P2 -p P002=S1 -p P003=S2 -i P000 -i P001 -i P002 -i P003 --sweep {fstart} {fstop} --sweep-stepsize {fstep} --verbose=3 --parallel=0 --simultaneous-frequencies=0 --recommended-memory --format=touchstone -s {os.getcwd()}/emx_s4p/{CellName}.s4p"
    os.system(emx_cmd)
    if os.path.exists(f"./{CellName}.log"):
        os.remove(f"./{CellName}.log")


def parse_freq_data(text):
    block_pattern = re.compile(
        r'(?i)"freq"\s+([\d.e+-]+)\s*((?:.*?(?=\s*"freq"|\s*END|\s*end|\Z))+)',
        re.DOTALL
    )
    blocks = []
    for match in block_pattern.finditer(text):
        freq = float(match.group(1))
        content = match.group(2).strip()
        blocks.append((freq, content))
    result = []
    for freq, content in blocks:
        current = {"freq": freq}
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # print(line)
            key=line.strip("\"").split("\"")[0].strip()
            valuestr=line.strip("\"").split("\"")[1].strip().strip("(").strip(")")
            value= complex(float(valuestr.split()[0]),float(valuestr.split()[1]))
            current[key]=value
        result.append(current)
    return result

def dBm2w(PindBm):
    return 10**(float(PindBm)/10) /1000

def exe_spectre_woxfmr(xnorm:list,Fre=2.4,RL=3e3 ):
    FWn     = int(xnorm[0]) * 10
    FWp     = int(xnorm[1]) * 10
    Fn_p    = int(xnorm[2])
    Lp      = int(xnorm[3]) * 10
    Mul_p   = int(xnorm[4])
    Fn_n    = int(xnorm[5])
    Ln      = int(xnorm[6]) * 10
    Mul_n   = int(xnorm[7])
    Cp      = int(xnorm[8])
    Cs      = int(xnorm[9])
    indp    = xnorm[10]
    inds    = xnorm[11]
    K       = xnorm[12]
    Pin=-5
    params_str = f"{FWn}_{FWp}_{Fn_p}_{Lp}_{Mul_p}_{Fn_n}_{Ln}_{Mul_n}_{Cp}_{Cs}_{indp}_{inds}_{K}_{Fre}_{RL}"
    hash_object = hashlib.md5(params_str.encode())
    hash_str = hash_object.hexdigest()
    short_hash_scs_filename = hash_str[:12]
    spectre_netlist=f"""
    simulator lang=spectre
    global 0
    parameters CP={Cp}f Cs={Cs}f Fn_n={Fn_n} Fn_p={Fn_p} Fre={Fre}G FWn={FWn}n FWp={FWp}n indp={indp}n \
        inds={inds}n K={K} Ln={Ln}n Lp={Lp}n Mul_n={Mul_n} Mul_p={Mul_p} Pin={Pin} RL={RL}
    include "/home/share/pdk/tsmc/tsmc28nm/T28HPC_RFPDK/tsmcN28/../models/spectre/toplevel.scs" section=top_tt
    
    subckt ideal_balun d c p n
        K0 (d 0 p c) transformer n1=2
        K1 (d 0 c n) transformer n1=2
    ends ideal_balun
    
    subckt RectifierCell_AutoOPT AVSS IN OUT RF_N RF_P
        M2 (net2 net1 IN AVSS) nch_lvt_mac l=Ln w=FWn*Fn_n multi=Mul_n nf=Fn_n \\
            sd=100n \\
            ad=((Fn_n-int(Fn_n/2)*2)*(7.5e-08+((Fn_n-1)*1e-07)/2+0)+(Fn_n+1-int((Fn_n+1)/2)*2)*((Fn_n/2)*1e-07))*FWn \\
            as=((Fn_n-int(Fn_n/2)*2)*(7.5e-08+((Fn_n-1)*1e-07)/2+0)+(Fn_n+1-int((Fn_n+1)/2)*2)*(7.5e-08+7.5e-08+(Fn_n/2-1)*1e-07+0+0))*FWn \\
            pd=(Fn_n-int(Fn_n/2)*2)*((7.5e-08+((Fn_n-1)*1e-07)/2+0)*2+(Fn_n+1)*FWn)+(Fn_n+1-int((Fn_n+1)/2)*2)*(((Fn_n/2)*1e-07)*2+Fn_n*FWn) \\
            ps=(Fn_n-int(Fn_n/2)*2)*((7.5e-08+((Fn_n-1)*1e-07)/2+0)*2+(Fn_n+1)*FWn)+(Fn_n+1-int((Fn_n+1)/2)*2)*((7.5e-08+7.5e-08+(Fn_n/2-1)*1e-07+0+0)*2+(Fn_n+2)*FWn) \\
            dfm_flag=0
        M7 (net1 net2 IN AVSS) nch_lvt_mac l=Ln w=FWn*Fn_n multi=Mul_n nf=Fn_n \\
            sd=100n \\
            ad=((Fn_n-int(Fn_n/2)*2)*(7.5e-08+((Fn_n-1)*1e-07)/2+0)+(Fn_n+1-int((Fn_n+1)/2)*2)*((Fn_n/2)*1e-07))*FWn \\
            as=((Fn_n-int(Fn_n/2)*2)*(7.5e-08+((Fn_n-1)*1e-07)/2+0)+(Fn_n+1-int((Fn_n+1)/2)*2)*(7.5e-08+7.5e-08+(Fn_n/2-1)*1e-07+0+0))*FWn \\
            pd=(Fn_n-int(Fn_n/2)*2)*((7.5e-08+((Fn_n-1)*1e-07)/2+0)*2+(Fn_n+1)*FWn)+(Fn_n+1-int((Fn_n+1)/2)*2)*(((Fn_n/2)*1e-07)*2+Fn_n*FWn) \\
            ps=(Fn_n-int(Fn_n/2)*2)*((7.5e-08+((Fn_n-1)*1e-07)/2+0)*2+(Fn_n+1)*FWn)+(Fn_n+1-int((Fn_n+1)/2)*2)*((7.5e-08+7.5e-08+(Fn_n/2-1)*1e-07+0+0)*2+(Fn_n+2)*FWn) \\
            dfm_flag=0
        M9 (net2 net1 OUT OUT) pch_lvt_mac l=Lp w=FWp*Fn_p multi=Mul_p nf=Fn_p \\
            sd=100n \\
            ad=((Fn_p-int(Fn_p/2)*2)*(7.5e-08+((Fn_p-1)*1e-07)/2+0)+(Fn_p+1-int((Fn_p+1)/2)*2)*((Fn_p/2)*1e-07))*FWp \\
            as=((Fn_p-int(Fn_p/2)*2)*(7.5e-08+((Fn_p-1)*1e-07)/2+0)+(Fn_p+1-int((Fn_p+1)/2)*2)*(7.5e-08+7.5e-08+(Fn_p/2-1)*1e-07+0+0))*FWp \\
            pd=(Fn_p-int(Fn_p/2)*2)*((7.5e-08+((Fn_p-1)*1e-07)/2+0)*2+(Fn_p+1)*FWp)+(Fn_p+1-int((Fn_p+1)/2)*2)*(((Fn_p/2)*1e-07)*2+Fn_p*FWp) \\
            ps=(Fn_p-int(Fn_p/2)*2)*((7.5e-08+((Fn_p-1)*1e-07)/2+0)*2+(Fn_p+1)*FWp)+(Fn_p+1-int((Fn_p+1)/2)*2)*((7.5e-08+7.5e-08+(Fn_p/2-1)*1e-07+0+0)*2+(Fn_p+2)*FWp) \\
            dfm_flag=0
        M8 (net1 net2 OUT OUT) pch_lvt_mac l=Lp w=FWp*Fn_p multi=Mul_p nf=Fn_p \\
            sd=100n \\
            ad=((Fn_p-int(Fn_p/2)*2)*(7.5e-08+((Fn_p-1)*1e-07)/2+0)+(Fn_p+1-int((Fn_p+1)/2)*2)*((Fn_p/2)*1e-07))*FWp \\
            as=((Fn_p-int(Fn_p/2)*2)*(7.5e-08+((Fn_p-1)*1e-07)/2+0)+(Fn_p+1-int((Fn_p+1)/2)*2)*(7.5e-08+7.5e-08+(Fn_p/2-1)*1e-07+0+0))*FWp \\
            pd=(Fn_p-int(Fn_p/2)*2)*((7.5e-08+((Fn_p-1)*1e-07)/2+0)*2+(Fn_p+1)*FWp)+(Fn_p+1-int((Fn_p+1)/2)*2)*(((Fn_p/2)*1e-07)*2+Fn_p*FWp) \\
            ps=(Fn_p-int(Fn_p/2)*2)*((7.5e-08+((Fn_p-1)*1e-07)/2+0)*2+(Fn_p+1)*FWp)+(Fn_p+1-int((Fn_p+1)/2)*2)*((7.5e-08+7.5e-08+(Fn_p/2-1)*1e-07+0+0)*2+(Fn_p+2)*FWp) \\
            dfm_flag=0
        C3 (net1 RF_N) capacitor c=2p
        C2 (RF_P net2) capacitor c=2p
    ends RectifierCell_AutoOPT
    
    PORT0 (net5 0) port r=50 type=sine freq=Fre dbm=Pin sinephase=0
    C26 (BALUN_P BALUN_N) capacitor c=CP
    C27 (RF_P RF_N) capacitor c=Cs
    C0 (Vout 0) capacitor c=1p
    I3 (net5 0 BALUN_P BALUN_N) ideal_balun
    RB (net2 RF_N) resistor r=0
    RT (RF_P net1) resistor r=0
    RLOAD (Vout 0) resistor r=RL
    I11 (0 net14 Vout net2 net1) RectifierCell_AutoOPT
    I0 (0 0 net14 net2 net1) RectifierCell_AutoOPT
    K0 mutual_inductor coupling=K ind1=L0 ind2=L3
    L3 (RF_P RF_N) inductor l=inds q=15 fq=2.4G mode=1
    L0 (BALUN_P BALUN_N) inductor l=indp q=15 fq=2.4G mode=1
    simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 \\
        iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \\
        maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \\
        sensfile="../psf/sens.output" checklimitdest=psf 
    hb  hb  autoharms=yes  autotstab=yes  oversample=[3]
    +   fundfreqs=[(Fre)]  maxharms=[5]  errpreset=conservative
    +   annotate=status
    modelParameter info what=models where=rawfile
    element info what=inst where=rawfile
    outputParameter info what=output where=rawfile
    designParamVals info what=parameters where=rawfile
    primitives info what=primitives where=rawfile
    subckts info what=subckts where=rawfile
    save Vout RLOAD:1 
    saveOptions options save=allpub
"""
    with open(f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs",mode="w") as netlist_file:
        netlist_file.write(spectre_netlist)
    spectre_cmd=f"source /home/imsic/zhangwx/EDA/env_config/env_spectre191_alpsnew_IC618new && spectre {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.scs -f psfascii"
    # os.system(spectre_cmd)
    subprocess.run(spectre_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open( f"{os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}.raw/hb.fd.pss_hb",mode="r") as psf:
        psf_content=psf.read()
    parsed_data = parse_freq_data(psf_content)
    eta_total = (parsed_data[0]["Vout"] * parsed_data[0]["RLOAD:1"]).real / dBm2w(Pin)
    os.system(f"rm -rf {os.getcwd()}/spectre_netlist_path/{short_hash_scs_filename}*")
    return eta_total



def find_max_spectre_rl(xnorm,Fre):
    left = 1 # note 1*500
    right = 200 #note: 200*500
    cache = {}
    max_val = -float('inf')
    best_rl = None
    def evaluate(rl):
        nonlocal max_val, best_rl
        if rl not in cache:
            val = exe_spectre_woxfmr(xnorm=xnorm,RL=rl*500,Fre=Fre)
            cache[rl] = val
            if val > max_val:
                max_val = val
                best_rl = rl
        return cache[rl]
    evaluate(left)
    evaluate(right)
    while right - left > 3:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        if m1 == left: m1 += 1
        if m2 == right: m2 -= 1
        if m1 >= m2: break
        v1 = evaluate(m1)
        v2 = evaluate(m2)
        if v1 < v2:
            left = m1
        else:
            right = m2
    for rl in range(left, right + 1):
        evaluate(rl)
    return best_rl, max_val



def round_to_nearest_100(x):
    return round(x / 100) * 100



def Rect_Opt_Flow(xnorm:list):
    try:
        print(f"Input to Rect_Opt_Flow: {xnorm}")
        RL_2_4G,best_eta_2_4G=find_max_spectre_rl(xnorm,Fre=2.4)
        # RL_5G, best_eta_5G = find_max_spectre_rl(xnorm, Fre=5)
        # FoM= 1/(best_eta_2_4G*100) + 1/(best_eta_5G*100)
        FoM = 1 / (best_eta_2_4G * 100)
        with open("result9.csv", mode="a+") as f:
            # f.write(
            #     f"{int(xnorm[0])}, {int(xnorm[1])},{int(xnorm[2])}, {int(xnorm[3])}, "
            #     f"" f"{int(xnorm[4])}, {int(xnorm[5])}, {int(xnorm[6])}, {int(xnorm[7])},"
            #     f" {int(xnorm[8])}, {int(xnorm[9])}, {int(xnorm[10])}, "
            #     f"{xnorm[11]},{xnorm[12]}, {RL_2_4G} , {best_eta_2_4G}, {RL_5G},{best_eta_5G} , {FoM} \n")

            f.write(
                f"{int(xnorm[0])}, {int(xnorm[1])},{int(xnorm[2])}, {int(xnorm[3])}, "
                f"" f"{int(xnorm[4])}, {int(xnorm[5])}, {int(xnorm[6])}, {int(xnorm[7])},"
                f" {int(xnorm[8])}, {int(xnorm[9])}, {int(xnorm[10])}, "
                f"{xnorm[11]},{xnorm[12]}, {RL_2_4G} , {best_eta_2_4G}, {FoM} \n")

    except:
        FoM=1e25
    os.system("rm -rf ./*.log")
    # with open("note.txt",mode="a+") as f2:
    #     f2.write(f"{str(xnorm)} {FoM}")
    return FoM if not (FoM == None) else 1e20



# print(Rect_Opt_Flow(
#     [260,9,4.7,2,2,
#      1000,1000,6,30,4,6,30,2,1000,1000]
# ))












