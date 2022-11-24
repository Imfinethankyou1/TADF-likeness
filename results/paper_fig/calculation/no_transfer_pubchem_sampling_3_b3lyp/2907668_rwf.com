%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/2907668_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/2907668_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.7313608010        1.6627650288       -0.6166422441
O                -4.9426358643        0.5236061214       -0.2944446512
C                -3.6125529780        0.6856766781       -0.0648065555
C                -2.9547507794        1.9305987544       -0.1253427193
C                -1.5953599669        2.0009757949        0.1249753737
C                -0.8618238001        0.8476108398        0.4349297389
C                 0.5804883891        0.9253028892        0.7179288458
O                 1.2261471978        1.9683701728        0.7160098716
C                 1.1950576988       -0.3850586372        1.0235779043
O                 2.4960523105       -0.4455125986        1.4286665692
C                 3.5038514808       -0.0539023152        0.5522467799
C                 4.6938040094        0.3711047917        1.1409220281
C                 5.7691970853        0.7175744172        0.3257136848
C                 5.6575583663        0.6422576195       -1.0652047815
C                 4.4605703570        0.2095261519       -1.6365189123
C                 3.3773548721       -0.1488306782       -0.8322037553
C                 0.4481289666       -1.5146378222        1.0275945320
C                 0.9933885585       -2.8903239169        1.3634553745
F                 2.1414519767       -3.1286820430        0.7057040288
F                 1.2392670510       -3.0119117970        2.6817115731
F                 0.1124517114       -3.8466127488        1.0199826800
O                -0.8838102428       -1.5411950002        0.7947705114
C                -1.5495381264       -0.3780053362        0.4898164286
C                -2.9213318215       -0.4990558190        0.2438379389
C                -3.6466626401       -1.8187123261        0.3021051372
H                -6.7468084376        1.2883236834       -0.7544734379
H                -5.7223249828        2.3994651674        0.1960797537
H                -5.3895409335        2.1378098487       -1.5444853274
H                -3.5038079524        2.8330216566       -0.3655329063
H                -1.0625134633        2.9454278399        0.0885153825
H                 4.7557877178        0.4285570770        2.2227131084
H                 6.6971094514        1.0524154214        0.7813350037
H                 6.4970186282        0.9172182276       -1.6971461828
H                 4.3647058860        0.1415834079       -2.7168449161
H                 2.4520593441       -0.5033085693       -1.2748040421
H                -4.1596304353       -2.0197349911       -0.6448859121
H                -2.9588424568       -2.6394482968        0.5091680882
H                -4.4181563770       -1.8042576935        1.0806830069


