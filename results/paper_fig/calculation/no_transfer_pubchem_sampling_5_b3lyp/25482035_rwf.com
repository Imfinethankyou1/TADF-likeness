%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/25482035_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/25482035_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.1478826901        0.8457174747       -0.6462006514
C                 4.8858921230        0.1168293607       -0.3150308060
C                 3.6423507802        0.5654025746        0.0471214417
C                 2.7913069857       -0.5773696798        0.2324217313
C                 1.3943867032       -0.6284226233        0.6327464513
O                 0.7862077327       -1.6826724759        0.8178901089
C                 0.6652412240        0.6896643828        0.9631658901
N                 0.5854301016        1.6932745469       -0.1117110487
C                 0.2015155776        3.0286724742        0.3678129783
S                -0.1991100493        1.1966260405       -1.5461146023
O                 0.4498869873       -0.0481092212       -1.9544437987
O                -0.2239731537        2.3752061942       -2.4167286721
C                -1.8975060312        0.8102766215       -1.0993673712
C                -2.9019356538        1.7918990016       -1.2943272429
C                -4.1981887872        1.5081813551       -0.9331379767
C                -4.5452224865        0.2524537803       -0.3637978649
C                -5.8754147513       -0.0650487308        0.0197219443
C                -6.1753845059       -1.2923183311        0.5670538813
C                -5.1598204823       -2.2622147249        0.7576075513
C                -3.8609047153       -1.9876280607        0.3953347366
C                -3.5188887351       -0.7302451761       -0.1725119056
C                -2.1867438317       -0.4244546960       -0.5560427443
C                 3.5617014656       -1.7135487774       -0.0315672375
C                 3.2121637319       -3.1653930690       -0.0098748616
N                 4.8114182728       -1.2716160840       -0.3539128725
H                 5.9871173716        1.9233086812       -0.5538395390
H                 6.9742988545        0.5754464347        0.0255788997
H                 6.4823840942        0.6499515228       -1.6740845611
H                 3.3522392984        1.6023044778        0.1381812571
H                 1.2199772050        1.1740410591        1.7764569871
H                -0.3257880831        0.4189590600        1.3544541489
H                 0.1793338876        3.7189111585       -0.4750100462
H                -0.7754346618        3.0354740938        0.8761012933
H                 0.9645098870        3.3643778721        1.0772264379
H                -2.6368051495        2.7395765969       -1.7498106276
H                -4.9803456848        2.2472580137       -1.0886448944
H                -6.6548401118        0.6787977744       -0.1268280037
H                -7.1971431362       -1.5227483339        0.8563725685
H                -5.4124069891       -3.2255862762        1.1916800657
H                -3.0755890705       -2.7249669584        0.5397732210
H                -1.4039469038       -1.1644527867       -0.4144859814
H                 2.3626571249       -3.3695052455       -0.6690307139
H                 4.0623152680       -3.7792825972       -0.3303497228
H                 2.9072887608       -3.4863173059        0.9918213149
H                 5.5778848462       -1.8807103979       -0.6036251621


