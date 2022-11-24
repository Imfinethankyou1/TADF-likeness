%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/108662373_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/108662373_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.7633863622       -0.8464390332       -2.7435708317
O                -3.1168754041       -0.1768881166       -1.6659733125
C                -3.7277826765       -0.1522177943       -0.4596708793
C                -5.0994832980       -0.3717770034       -0.2748511277
C                -5.6793804966       -0.2454567719        0.9869524757
C                -4.8826616292        0.1084576709        2.0669398923
F                -5.4383753626        0.2352605252        3.2913207293
C                -3.5213116669        0.3236955186        1.9176022228
C                -2.9213374983        0.1696667207        0.6570307669
C                -1.4637039509        0.3960455980        0.5515430389
O                -1.0043858868        1.5332094337        1.1474194849
C                -0.5441057918       -0.4404656918        0.0094301345
C                -0.7786597861       -1.8019235000       -0.4722869553
O                -1.8137889730       -2.4359752415       -0.5818899800
C                 0.6068062335       -2.3801051110       -0.8163245928
O                 0.8467575214       -3.4996667573       -1.2310124961
N                 1.5159264569       -1.3832593995       -0.5647435263
C                 2.9488148383       -1.5910050971       -0.7192115192
C                 3.7164982688       -1.4476744298        0.5550475379
C                 4.7407168162       -0.6312996676        0.9408868260
C                 5.0808537152       -1.0059561448        2.2837286030
C                 4.2331909710       -2.0169507767        2.6155803820
O                 3.3967476453       -2.3006130263        1.5766728533
C                 0.9465792061       -0.1241783510       -0.0596013240
C                 1.3119478441        1.0634275181       -0.9201457142
S                 0.9653838900        1.0615388414       -2.6366852590
C                 1.6302640575        2.6504705030       -2.8184854830
C                 2.0841125699        3.1429050998       -1.6287873005
C                 1.9093496790        2.2402714817       -0.5283835530
C                 2.3191072533        2.5760810552        0.8849683703
H                -3.0320910640       -0.8741222438       -3.5523185418
H                -4.6531402187       -0.2994304485       -3.0816731456
H                -4.0330648419       -1.8687979070       -2.4586095364
H                -5.7265610575       -0.6332678458       -1.1186329339
H                -6.7407031269       -0.4157466779        1.1350268662
H                -2.9211263478        0.5535383142        2.7926464798
H                -1.7625313784        2.1068328681        1.3486527893
H                 3.3559229100       -0.8805325759       -1.4478531518
H                 3.0487018796       -2.6018928709       -1.1291620112
H                 5.2034877115        0.1371467501        0.3367457728
H                 5.8501021647       -0.5769285732        2.9106005038
H                 4.0945474698       -2.6150400237        3.5033768945
H                 1.3437930385        0.0727915272        0.9449085711
H                 1.6539513267        3.1209647174       -3.7923605272
H                 2.5403443507        4.1222594813       -1.5222908008
H                 2.8777760044        3.5170738897        0.9102888343
H                 1.4445899039        2.6864633323        1.5363561091
H                 2.9581850921        1.7995122338        1.3207993644


