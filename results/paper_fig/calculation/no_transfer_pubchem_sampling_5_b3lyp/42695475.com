%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/42695475.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.6646680232        4.3457239650        1.3433254266
O                -5.6853648370        2.9399912630        1.3116470907
C                -4.8680208808        2.2955519990        0.4331960562
C                -3.9864113860        2.9185586731       -0.4412841693
C                -3.2013268716        2.1540384166       -1.2901853960
C                -3.2826413005        0.7719057669       -1.2936322861
C                -2.4158392715       -0.0696134415       -2.1901617859
N                -1.4807993869       -0.8628289941       -1.4080134848
C                -0.4045846179       -0.1533843770       -0.7577110689
C                 0.8417639474       -0.0332981469       -1.6547014374
C                 2.0198912567        0.4228442811       -0.8441457576
C                 2.4462377593        1.7348222059       -0.8520058513
C                 3.5208428337        2.1351821173       -0.0674145516
C                 4.1790715057        1.2282084294        0.7424387467
O                 5.2403351976        1.5145563105        1.5477532182
C                 5.7258970082        2.8325652362        1.5649146135
C                 3.7446532778       -0.1173096729        0.7627616927
O                 4.4425873916       -0.9463701953        1.5870261695
C                 4.0549713520       -2.2949940115        1.6577554019
C                 2.6769229899       -0.4975226235       -0.0275533796
C                -1.6117154726       -2.2088699011       -1.3857608577
O                -2.4919958259       -2.7994139222       -1.9855039919
C                -0.5956682573       -2.9508207393       -0.5675309200
C                 0.4529746826       -3.6060160472       -1.1997212742
C                 1.3683254235       -4.3316105855       -0.4564868803
C                 1.2344656009       -4.4168807522        0.9191411460
C                 0.1784590411       -3.7804820004        1.5509325237
C                -0.7357172565       -3.0503804616        0.8118717159
C                -4.1672097886        0.1538250118       -0.4134368395
C                -4.9485942018        0.9007190781        0.4385289035
H                -4.6787114228        4.7279406643        1.6279409548
H                -5.9562337106        4.7744543713        0.3789135516
H                -6.3931296816        4.6344421675        2.0993617753
H                -3.8980712455        3.9931719632       -0.4731298594
H                -2.5213395872        2.6544237693       -1.9658045766
H                -1.8591392160        0.5598669957       -2.8911501947
H                -3.0278724724       -0.7805589673       -2.7526312108
H                -0.7661417511        0.8418156944       -0.4865445332
H                -0.1259797416       -0.6723253520        0.1614006704
H                 0.6466436578        0.6655773606       -2.4689940755
H                 1.0507100793       -1.0144066148       -2.0854360532
H                 1.9497607105        2.4629801615       -1.4757680610
H                 3.8339085908        3.1664025862       -0.1002341611
H                 4.9666457422        3.5387522813        1.9182697166
H                 6.0797649888        3.1459904076        0.5767559942
H                 6.5631197238        2.8278804913        2.2610579116
H                 4.1502937298       -2.7947496956        0.6878403541
H                 4.7359985704       -2.7579830064        2.3701182025
H                 3.0253255549       -2.4025884064        2.0155976737
H                 2.3360861832       -1.5214511912       -0.0284050456
H                 0.5453530653       -3.5520435287       -2.2749063982
H                 2.1847812801       -4.8354804324       -0.9529910130
H                 1.9480782004       -4.9852405230        1.4976235798
H                 0.0654381582       -3.8550891605        2.6226406948
H                -1.5637816137       -2.5582758696        1.3021327415
H                -4.2389969579       -0.9236315432       -0.4095721303
H                -5.6393527251        0.4314294954        1.1209347185


