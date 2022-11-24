%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/73013451_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/73013451_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.3401113872       -2.4871581970       -0.2154486994
C                -1.2864045827       -1.0419911218       -0.7439318376
C                -2.5568381853       -0.3008459362       -0.3378742660
C                -2.7161137078        0.1867943440        0.9666596964
C                -3.9011993716        0.8127520868        1.3523535559
C                -4.9496761208        0.9582374543        0.4403183356
C                -4.8006043824        0.4775567412       -0.8609323322
C                -3.6105249765       -0.1444539578       -1.2448657118
N                -0.0976255275       -0.3359339018       -0.2625034604
C                 0.0622268256        1.0062081255       -0.8492667413
C                 1.5342730923        1.3884551178       -0.5783773999
C                 2.2181666786        0.0701661794       -0.0816102199
C                 2.4885616419        0.1047089042        1.4376509333
C                 3.8474214988        0.8138179962        1.5294168172
C                 4.6038412306        0.2689781311        0.3002933032
N                 3.5553249186       -0.2142576717       -0.6235053131
C                 1.1873981061       -0.9952018274       -0.4941052907
H                -2.2827754371       -2.9554619604       -0.5153470568
H                -1.2860756559       -2.4976139791        0.8786950027
H                -0.5209913689       -3.0958194533       -0.6115144354
H                -1.2701488132       -1.0942347172       -1.8533168842
H                -1.8928629575        0.0832654611        1.6678419467
H                -4.0071927143        1.1889741960        2.3669477948
H                -5.8730198074        1.4464638422        0.7410682095
H                -5.6072981500        0.5913128249       -1.5807484262
H                -3.4977162147       -0.5117619882       -2.2630280405
H                -0.1304668025        0.9836094226       -1.9396871393
H                -0.6575282155        1.7030011106       -0.4108131188
H                 1.6231504918        2.1927303224        0.1599710254
H                 2.0148577484        1.7455510981       -1.4965285554
H                 2.5722151954       -0.9211239446        1.8178482785
H                 1.6890069549        0.6032054539        1.9918472304
H                 3.7138103301        1.8994391481        1.4446842979
H                 4.3720431444        0.6211764042        2.4708258924
H                 5.2619193564       -0.5645619150        0.5845394774
H                 5.2421934922        1.0357551609       -0.1588299387
H                 3.6694051714        0.1505861861       -1.5637277487
H                 1.2910180896       -1.9089427627        0.0959198866
H                 1.3383384115       -1.2633813771       -1.5609190673


