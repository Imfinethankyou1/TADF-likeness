%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/114977603_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/114977603_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.0077276517        3.9488711496        0.2075339839
C                -1.2236937600        2.7853738681        0.4508568943
C                -1.5034223663        1.3757767194        0.7201320683
C                -0.8491033621        0.3737331127       -0.2867688998
N                -1.4513819462       -0.9583132171       -0.2635302869
C                -2.8028805490       -1.0829791052       -0.8054581384
C                -3.2455119465       -2.5448174980       -0.8075967155
C                 0.6409959649        0.2305662374       -0.0700560944
C                 1.5411053960        1.1184837691        0.4583451910
C                 2.8703617828        0.6061770567        0.4209169372
S                 4.4339938709        1.1953527469        0.9301308719
C                 5.1938551015       -0.2867608801        0.3869941143
C                 4.3084037022       -1.1791169203       -0.1492934016
C                 2.9781909418       -0.6638036484       -0.1322246904
S                 1.4120820378       -1.2643850467       -0.6173145047
H                -0.8114433191        4.9747199032       -0.0073167228
H                -1.1820109238        1.1133081827        1.7370701702
H                -2.5898257265        1.2252305531        0.6834482679
H                -1.0262118384        0.7688356877       -1.2973895280
H                -1.3989975936       -1.3535776131        0.6747096499
H                -3.5532115847       -0.4767633462       -0.2645875466
H                -2.7798270128       -0.7009880341       -1.8341539539
H                -4.2436502494       -2.6457089514       -1.2472409327
H                -2.5437455308       -3.1579724889       -1.3814243455
H                -3.2907926674       -2.9459835331        0.2127734297
H                 1.2649290145        2.0967315004        0.8320210208
H                 6.2638719062       -0.4028846480        0.4969727159
H                 4.5956493095       -2.1491055571       -0.5387945542


