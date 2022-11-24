%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/124089979.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.7831266461       -0.2406540627       -2.0384424174
N                -0.8495129477        0.1719037568       -1.0106285890
C                -0.9470257456        1.4674743331       -0.5580450552
C                -2.3347352188        1.8902245514       -0.2437522054
C                -3.1601136763        1.0446984043        0.4960665535
C                -4.4390391325        1.4317673590        0.8410368522
C                -4.9122403845        2.6733819177        0.4337259410
C                -6.2961926748        3.0960698624        0.8307446452
F                -6.3330136857        3.6040670733        2.0800253830
F                -7.1705492426        2.0747926666        0.8235932602
F                -6.8100933532        4.0405620170        0.0294148111
C                -4.1032447741        3.5270274717       -0.3045416854
C                -2.8218614605        3.1330857513       -0.6395437334
C                 0.0317402102        2.3715834768       -0.3324238732
C                 1.4524115346        2.2936818853       -0.6695818759
O                 2.0691787171        3.2719826042       -1.0630729419
N                 2.0308913748        1.0669987555       -0.5635723535
C                 3.4228905560        0.9211801908       -0.8954942868
C                 4.3199492753        0.8177619332        0.3065508661
N                 3.7702455672        0.8975875308        1.5099659762
C                 4.5840885039        0.8507473264        2.5514822883
C                 5.9565635276        0.7156001625        2.4117805612
C                 6.4234857709        0.6070403321        1.1085013236
N                 5.6155194517        0.6537008946        0.0652018758
C                 1.3463426368       -0.0297842009        0.0687366341
C                 0.3278648868       -0.6754484786       -0.8762848013
C                -0.1398119365       -2.0907269800       -0.4071479050
C                -0.2101736789       -3.0479136218       -1.5686138703
C                -0.0092387269       -4.3510972077       -1.4642927406
C                 0.3134795122       -5.0488579530       -0.1816026100
C                 0.1182641089       -4.1222620856        1.0144359809
C                 0.6917925220       -2.7411757019        0.7041880092
H                -2.5603086995        0.5093559177       -2.1599816094
H                -1.2629448345       -0.3484826846       -2.9976883714
H                -2.2585498055       -1.1917525984       -1.7949546238
H                -2.7859878562        0.0827220791        0.8113318631
H                -5.0741455830        0.7778448847        1.4190261445
H                -4.4798172922        4.4880541821       -0.6193876557
H                -2.1943574910        3.7876959452       -1.2244142230
H                -0.2626586343        3.3431586575        0.0344864619
H                 3.5831999522        0.0499479656       -1.5407334646
H                 3.7116757452        1.8193449858       -1.4535117425
H                 4.1091730231        0.9213740709        3.5221182395
H                 6.6185836936        0.6877371476        3.2609836512
H                 7.4750277838        0.4773535798        0.8852329527
H                 2.1031959910       -0.7573591535        0.3494991537
H                 0.8431885584        0.3147428004        0.9767444369
H                 0.8177253788       -0.7816978410       -1.8585529978
H                -1.1510317710       -1.9617647711        0.0003276676
H                -0.4333592548       -2.6215291833       -2.5357425053
H                -0.0587792081       -4.9793302468       -2.3432706488
H                 1.3531286231       -5.3910546430       -0.2249741713
H                -0.3101118382       -5.9405276322       -0.0801279010
H                 0.6079915405       -4.5412853640        1.8951934830
H                -0.9468117072       -4.0303516965        1.2364389929
H                 0.6746657954       -2.1262818205        1.6043646628
H                 1.7265730193       -2.8629145459        0.3791831875


