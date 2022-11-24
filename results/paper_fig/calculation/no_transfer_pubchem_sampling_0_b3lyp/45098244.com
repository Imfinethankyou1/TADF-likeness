%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/45098244.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.1159558112        1.5654890878       -0.1979765270
C                -1.1293317628        0.4160319073       -0.1866179160
C                -1.6316261148       -0.8454801995        0.5594382616
C                -0.7989603422       -1.9878562916       -0.0528014765
C                 0.1699949364       -1.2474121438       -0.9831099340
C                -0.7630138235       -0.1806481717       -1.5547005906
C                 1.1615425238       -0.3982936556       -0.1545686042
C                 2.2458467815        0.2018414567       -1.0598616303
C                 1.8607426748       -1.1706119234        0.9693900665
C                 0.2390993738        0.6782394745        0.3836547742
C                 0.6257028177        1.6260724659        1.2403434416
C                -0.1834902775        2.6291687700        1.7959643024
N                -0.7824588055        3.4727034702        2.2950591795
H                -2.4198502444        1.8288060541        0.8116087138
H                -1.6763891489        2.4448545742       -0.6641000406
H                -3.0030196695        1.2829652692       -0.7619907212
H                -1.4867441691       -0.7503479263        1.6356729631
H                -2.6970889948       -0.9826560775        0.3760249552
H                -0.2937009346       -2.5934477234        0.6953110297
H                -1.4300873231       -2.6540348394       -0.6424809366
H                 0.6638267290       -1.8959756159       -1.7054991452
H                -0.2798526263        0.5373064920       -2.2136684984
H                -1.6265912088       -0.6039023606       -2.0647702917
H                 2.8621725257        0.9023128409       -0.5007066129
H                 2.8864222440       -0.5909490696       -1.4406050411
H                 1.8177580385        0.7329017592       -1.9052364668
H                 2.2681843147       -2.1024705121        0.5830058850
H                 2.6830133164       -0.5847638309        1.3738997609
H                 1.1802654227       -1.3951782844        1.7861360527
H                 1.6535895578        1.6653340039        1.5831840467


