%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/671152_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/671152_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -0.2205621889       -3.3928013346        0.9019670008
C                -1.2919582280       -2.8466539387        0.7610391851
O                -2.4831629970       -3.4991023930        1.0102874983
N                -3.5801216965       -2.6410847031        0.7444172358
C                -3.1073636260       -1.4854106839        0.3636752640
C                -4.0222469940       -0.3990994850        0.0379827098
C                -3.8594966954        0.6264169756       -0.8680164175
C                -4.9888971603        1.4895288478       -0.9459654916
C                -5.9978944980        1.1115879181       -0.1009983334
S                -5.5927709740       -0.3078561129        0.8021370806
C                -1.6540547997       -1.4742063303        0.3507606746
C                -0.7869378439       -0.4445905801        0.1255526313
C                 0.6402016437       -0.5388139794        0.1739898761
C                 1.4272144579        0.5477708997       -0.0494613043
C                 2.8816484331        0.6057511348       -0.0323549176
C                 3.6928355791       -0.5173063162        0.2356698825
C                 5.0779335139       -0.4045189570        0.2392859394
C                 5.6883672490        0.8270779980       -0.0233323043
C                 4.9011271373        1.9487571991       -0.2903082881
C                 3.5135244725        1.8378050662       -0.2944345789
H                -2.9783934669        0.7352814376       -1.4898347260
H                -5.0477264809        2.3480977152       -1.6057280323
H                -6.9597512401        1.5857265640        0.0429265693
H                -1.2087943391        0.5356348705       -0.0919644147
H                 1.0624439767       -1.5130271397        0.3971454212
H                 0.9305195761        1.4942149600       -0.2684986094
H                 3.2355563312       -1.4798601259        0.4421725919
H                 5.6874804036       -1.2792349905        0.4478453709
H                 6.7716699095        0.9093672771       -0.0190030252
H                 5.3680758352        2.9081177792       -0.4947181264
H                 2.9015337102        2.7124304273       -0.5022383619


