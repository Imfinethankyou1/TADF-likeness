%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/41403978_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/41403978_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -0.5427459027       -3.2245319554       -1.3340411701
C                -1.4074229187       -2.4914234161       -1.8038291100
C                -1.3887978564       -2.0294719921       -3.2650291859
S                 0.1654254508       -1.1466005021       -3.7216449571
C                 0.2630842890        0.0273341832       -2.4071412623
N                -0.6374574747        0.9288652161       -2.1050795508
N                -0.1888752147        1.6165938548       -0.9941246304
C                 0.9958648631        1.1310301879       -0.6695662248
N                 1.8172540626        1.6224645004        0.3248036919
C                 2.3396038999        0.7038789412        1.3523641759
C                 3.6996674696        1.2203874147        1.8612302593
C                 3.7601333469        2.7638037578        1.8109942014
C                 2.3597659938        3.3557170803        2.0097995473
C                 1.4089187276        2.9301029354        0.8714224908
N                 1.3437950907        0.1067694616       -1.5318821569
C                 2.5825214618       -0.6136691845       -1.5964942400
C                 3.7656517863        0.0879431911       -1.8470606921
C                 4.9746929202       -0.6055216303       -1.9040400377
C                 4.9975871199       -1.9902234093       -1.7228740706
C                 3.8085257582       -2.6816550140       -1.4824805090
C                 2.5926933420       -1.9997675748       -1.4128868901
N                -2.4695880185       -1.9858904952       -1.1141545430
C                -2.7738126587       -2.1853158196        0.2475843901
C                -2.5149900648       -3.4098483527        0.8793858023
C                -2.8257908409       -3.6000000905        2.2221114222
C                -3.4210815822       -2.5714242450        2.9523723128
C                -3.6879158626       -1.3533759345        2.3340849892
C                -3.3646773726       -1.1333074859        0.9878424763
N                -3.5667044151        0.1733184964        0.4497432317
C                -3.2704182714        1.3706878660        1.2595103745
C                -3.2902565569        2.5161626390        0.2327102930
C                -4.2490213528        2.0080296509       -0.8521424864
C                -4.1572349040        0.4937655454       -0.7505302874
O                -4.5538941197       -0.3106883054       -1.5932869200
H                -1.3956026175       -2.9098045609       -3.9155455429
H                -2.2444842998       -1.3974241758       -3.5108768548
H                 1.6150173013        0.6282415046        2.1803642690
H                 2.4382304382       -0.2959293517        0.9290094290
H                 4.5072077673        0.7960097278        1.2549044120
H                 3.8470830652        0.8628849348        2.8879594183
H                 4.1490614006        3.0864686773        0.8375497509
H                 4.4528780479        3.1452784368        2.5697281858
H                 2.3940676120        4.4501291560        2.0526316964
H                 1.9617931954        3.0223955940        2.9773317962
H                 1.4382161535        3.6510133261        0.0478163997
H                 0.3692919316        2.8996218346        1.2299656781
H                 3.7310980395        1.1628131784       -1.9914887998
H                 5.8959819129       -0.0637579030       -2.0993148870
H                 5.9399494935       -2.5288229320       -1.7729078350
H                 3.8218915812       -3.7593684574       -1.3468969167
H                 1.6607109713       -2.5287218449       -1.2356540676
H                -3.1510500788       -1.4062881454       -1.6099890043
H                -2.0454027358       -4.1963102328        0.3031468311
H                -2.6086985369       -4.5546190995        2.6927886051
H                -3.6803068136       -2.7130255679        3.9976574256
H                -4.1536775254       -0.5546663516        2.9019832777
H                -4.0349768307        1.5083808966        2.0366698475
H                -2.2999934976        1.2516012394        1.7493597845
H                -3.6046643160        3.4614288327        0.6843191088
H                -2.2867511136        2.6353222511       -0.1850027643
H                -3.9892020042        2.3239753802       -1.8650476871
H                -5.2921687361        2.2990341378       -0.6680612898


