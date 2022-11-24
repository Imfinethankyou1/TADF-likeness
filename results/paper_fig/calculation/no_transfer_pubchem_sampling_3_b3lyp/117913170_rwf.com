%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/117913170_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/117913170_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 9.4658939407       -1.0998147985       -1.5580380899
C                 8.1954346032       -0.9968743034       -1.1594066360
C                 7.7176284839        0.2235429675       -0.4739575846
O                 8.3729314029        1.2078776176       -0.2095478501
O                 6.3874552301        0.0911545885       -0.1653178136
C                 5.7800774701        1.1456571637        0.4797160899
C                 4.4896111926        1.0897527358        0.8125197012
O                 3.7294711106       -0.0316872062        0.6100143737
C                 2.3761602734        0.1187184060        0.3999695287
C                 1.8330480246        1.2732795849       -0.2169313786
C                 0.4762894030        1.3618672919       -0.4317488818
C                -0.3857706797        0.3019253281       -0.0525252428
C                -1.7863594891        0.3609688289       -0.2512941958
F                -2.3006232798        1.4842170179       -0.8052890397
C                -2.6226070276       -0.6775775759        0.1014757721
O                -3.9891411750       -0.6511312892       -0.0376353627
C                -4.5331694024       -0.1529208467       -1.1973993953
C                -5.8174226773        0.1925006040       -1.2769421336
O                -6.6318478038        0.1243979609       -0.1655977369
C                -7.9424632838        0.4836974196       -0.3203275144
O                -8.4191386550        0.8483395728       -1.3742023219
C                -8.7042863180        0.3749401055        0.9424973933
C                -8.1888542330       -0.0187821406        2.1105824551
C                -2.0634768862       -1.8285129521        0.7109465900
C                -0.7121706793       -1.9228457219        0.9379570069
C                 0.1669499772       -0.8701020046        0.5595160536
C                 1.5643635069       -0.9305111730        0.7761825335
H                10.1662886041       -0.2879533900       -1.3841326603
H                 9.8339990520       -1.9878582985       -2.0623623380
H                 7.4682340304       -1.7874600329       -1.3161687059
H                 6.4089696232        1.9965287946        0.7048893060
H                 4.0137209875        1.9200420524        1.3264116960
H                 2.4932646526        2.0722528819       -0.5381557223
H                 0.0547844536        2.2414360856       -0.9059582741
H                -3.8847763026       -0.1026208805       -2.0644756878
H                -6.2809176934        0.5330277532       -2.1925331441
H                -9.7492943082        0.6516045848        0.8394110206
H                -7.1431949723       -0.2927333324        2.2060616149
H                -8.8046989497       -0.0758634079        3.0035190818
H                -2.7435932523       -2.6273583280        0.9888384499
H                -0.2993164672       -2.8121904873        1.4054267713
H                 2.0085495134       -1.8036961770        1.2440142716


