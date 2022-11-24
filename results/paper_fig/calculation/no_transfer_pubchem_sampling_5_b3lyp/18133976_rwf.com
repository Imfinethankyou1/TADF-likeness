%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/18133976_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/18133976_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -4.3468979353       -0.6096209681       -1.5054164452
S                -3.1270617865       -1.4342364668       -1.4335127203
O                -2.7726988750       -2.3398332839       -2.5384079707
C                -3.1420724708       -2.3451220287        0.1566323053
C                -2.5158312218       -1.3642402289        1.1484282624
C                -1.2722971952       -0.7702800918        0.4477241026
N                -0.5611007596        0.2311565176        1.2438523412
C                -1.3428413143        1.2945467898        1.8988676730
C                -2.0757865783        2.2731440370        1.0277345392
C                -1.7569479492        3.4663164150        0.4445950680
C                -2.8878810793        3.8566065946       -0.3473664138
C                -3.8170285488        2.8748094771       -0.1901336767
O                -3.3436606439        1.9078047938        0.6474090436
C                 0.6715502659        0.7079858184        0.6077475547
C                 1.7855634240       -0.3335461057        0.5768589177
C                 1.8367252400       -1.3080074298        1.5545382512
C                 2.8681771775       -2.2740243608        1.5889746563
C                 3.8595843427       -2.2585964713        0.6379256311
C                 3.8647460041       -1.2688725285       -0.3801845010
C                 4.8918672647       -1.2344538226       -1.3611754115
C                 4.9097161588       -0.2683064375       -2.3399942060
C                 3.8919213835        0.7124519173       -2.3755854049
C                 2.8788496586        0.7056259932       -1.4419156271
C                 2.8194844605       -0.2823356486       -0.4174661124
C                -1.7061574030       -0.3537508617       -0.9950269446
H                -4.1782145567       -2.6120681153        0.3744941330
H                -2.5425631484       -3.2496176575        0.0189269192
H                -3.2453751910       -0.5832540952        1.3802899800
H                -2.2315219762       -1.8598781136        2.0823122733
H                -0.5624058582       -1.5964915477        0.3220522774
H                -0.6401858725        1.8496843268        2.5288560918
H                -2.0624367952        0.8260630399        2.5772620921
H                -0.8287472373        4.0083293270        0.5663728196
H                -2.9914970112        4.7498987385       -0.9474464498
H                -4.8085852249        2.7077715106       -0.5813202505
H                 1.0217047341        1.5730346768        1.1866037252
H                 0.4783826411        1.0913045798       -0.4048373963
H                 1.0540681840       -1.3233593448        2.3070931933
H                 2.8692844588       -3.0276238231        2.3719959339
H                 4.6561232638       -2.9986620849        0.6523873035
H                 5.6717115974       -1.9913784362       -1.3200945470
H                 5.7020992421       -0.2542385232       -3.0832360577
H                 3.9095201900        1.4787843547       -3.1458663124
H                 2.1199487580        1.4789560406       -1.4913469133
H                -2.0718225053        0.6705766760       -1.0588679358
H                -0.9294103115       -0.5230541482       -1.7428897915


