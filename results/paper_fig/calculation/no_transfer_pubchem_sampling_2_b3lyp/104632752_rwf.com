%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/104632752_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/104632752_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                 3.9171159171        1.3642593795       -1.5594402379
C                 3.9089018223        0.4547477630       -0.4595826449
C                 2.5221693538        0.2381198210        0.1226917674
C                 2.0807743576        1.1013303994        1.1313953542
C                 0.7934231888        0.9942549773        1.6567902800
C                -0.0940574531        0.0182912052        1.1909554019
C                -1.4884902005       -0.1198160242        1.7810729267
C                -2.6139894143       -0.0669499383        0.7602173063
C                -3.5080640092       -1.1340673972        0.6190808635
C                -4.5512937613       -1.0792423997       -0.3089642247
C                -4.7141045442        0.0493382202       -1.1113114334
C                -3.8267486195        1.1212725030       -0.9804687859
C                -2.7866888743        1.0615129600       -0.0548178296
C                 0.3483303932       -0.8443673286        0.1789488538
C                 1.6324982860       -0.7393959725       -0.3497116117
C                 4.6530585737       -0.8086828602       -0.8902941465
F                 4.0506220647       -1.3526026547       -2.0065330097
F                 4.6372891934       -1.7502032093        0.0944727114
H                 3.3438974822        0.9795770902       -2.2431394229
H                 4.5329299699        0.9331747894        0.3028833448
H                 2.7507136369        1.8710340185        1.5079618365
H                 0.4773705262        1.6753736527        2.4438560902
H                -1.5561121692       -1.0679189786        2.3309350718
H                -1.6335558725        0.6724865907        2.5271105925
H                -3.3895836180       -2.0164448501        1.2444475541
H                -5.2345183624       -1.9194642869       -0.4026237387
H                -5.5245642743        0.0954823068       -1.8338175970
H                -3.9456391421        2.0052437621       -1.6018303553
H                -2.0951837621        1.8960612619        0.0355873026
H                -0.3248458609       -1.6072090303       -0.2044107681
H                 1.9465948161       -1.4318344054       -1.1222174250
H                 5.6917503559       -0.5933613644       -1.1605390258


