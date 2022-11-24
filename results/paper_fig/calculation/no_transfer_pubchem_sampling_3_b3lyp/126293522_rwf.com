%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/126293522_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/126293522_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.7827720176        2.9388251177        0.0169075655
O                -5.1277146638        2.0452033227        0.9052878370
C                -4.0493827154        1.3385369747        0.4566636310
C                -3.5230110612        1.4717831521       -0.8301841280
C                -2.4150495715        0.7293150853       -1.2577598230
C                -1.9307427756        0.9110760851       -2.6788423125
C                -1.8156906950       -0.1504811924       -0.3337429762
C                -0.6846624415       -1.0259124131       -0.7410108375
N                -0.7830533027       -1.7247034130       -1.8306296319
C                 0.2483705693       -2.5589191594       -2.2020188730
C                 0.1316786461       -3.2954787821       -3.3992164823
C                 1.1429093207       -4.1590717793       -3.7832886535
C                 2.2955565158       -4.3190192875       -2.9884141930
C                 2.4271971849       -3.6078350409       -1.8085716696
C                 1.4079079596       -2.7226081312       -1.4098314074
C                 1.5338436781       -1.9527231507       -0.1849124776
O                 2.4769516140       -2.0291586280        0.6130721245
N                 0.4781923944       -1.0568079211        0.0502755937
N                 0.4835215699       -0.3662915873        1.2958643946
C                 1.3943115547        0.5226288269        1.4532168078
C                 2.3883763007        1.0830849066        0.5146184809
C                 2.0355520293        1.4732542297       -0.7813211136
C                 2.9798209664        2.1026385267       -1.5855486578
N                 2.5922820920        2.5188049043       -2.9406798381
O                 3.4391736486        3.0883225434       -3.6262865509
O                 1.4426979338        2.2736226721       -3.3104670661
C                 4.2770599067        2.3622289018       -1.1419925430
C                 4.6326892455        1.9803601965        0.1459542468
C                 3.6971745230        1.3522290275        0.9739515304
O                 4.0687359501        1.0393502127        2.2545412550
C                 4.3372678792       -0.3527130509        2.5169061589
C                 5.4845339339       -0.8700372453        1.6245101741
O                 6.5795143199       -0.3588559476        1.6468614936
O                 5.2002830854       -1.9145095297        0.8338687593
C                -2.3642429337       -0.2785139351        0.9569154433
C                -3.4759970192        0.4384139599        1.3824994022
C                -4.0611646938        0.3078784253        2.7836508982
C                -3.7725817186       -1.0449879519        3.4507722981
C                -3.5883151626        1.4694800135        3.6833695350
H                -5.1129563192        3.7439761404       -0.3110366356
H                -6.6111824841        3.3669341614        0.5840261281
H                -6.1765015632        2.4144761435       -0.8630159582
H                -3.9751512412        2.1626563397       -1.5325750446
H                -2.1714439034        0.0343465453       -3.2879465645
H                -0.8469327850        1.0497316944       -2.7434851049
H                -2.4033553189        1.7877264095       -3.1324481051
H                -0.7651769609       -3.1632727452       -3.9955389675
H                 1.0474457151       -4.7215427123       -4.7080268848
H                 3.0806914240       -4.9997526137       -3.3033445964
H                 3.3080532152       -3.7121113307       -1.1840179363
H                 1.4128738267        0.9397311134        2.4611304535
H                 1.0309415674        1.3297158171       -1.1554916015
H                 4.9818198680        2.8484101730       -1.8050891326
H                 5.6333669081        2.1410620934        0.5302722421
H                 4.6744744949       -0.3802055111        3.5549192530
H                 3.4260622223       -0.9446260773        2.4063527000
H                 4.2310726315       -2.0920750628        0.7974799779
H                -1.9012653247       -0.9749530424        1.6440220096
H                -5.1491534154        0.4020848728        2.6814020249
H                -4.3231848755       -1.1185010336        4.3953285438
H                -2.7087810331       -1.1676382317        3.6871314903
H                -4.0778186780       -1.8837793371        2.8152284970
H                -3.8466262438        2.4381488578        3.2452629094
H                -2.5008059535        1.4342048699        3.8217896672
H                -4.0576878229        1.4068445284        4.6726832400


