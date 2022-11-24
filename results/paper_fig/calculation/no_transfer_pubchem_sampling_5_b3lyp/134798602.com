%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/134798602.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.3816476577        0.1610106556        0.2887578949
C                 3.7274726514       -0.4366653440       -0.9611371992
C                 4.6069108364       -1.5250706349       -1.5672232025
C                 2.3795235642       -1.0256232149       -0.5581136800
O                 2.1935858712       -2.1912452883       -0.3104034310
N                 1.3978412989       -0.0728232455       -0.4631039703
C                 0.0809211887       -0.3217731581       -0.1141881958
C                -0.6212094641       -1.4991562690       -0.0159969741
C                -1.9468216725       -1.1027763687        0.2769153448
C                -3.1121324962       -1.9435634883        0.4874440680
C                -4.3589691561       -1.3722163082        0.7484466066
C                -5.4445432964       -2.2064003395        0.9411485684
N                -5.3715283198       -3.5274635213        0.8925377428
C                -4.1902859878       -4.0693950076        0.6469941926
C                -3.0354423434       -3.3353558312        0.4380924069
N                -2.0326440859        0.2185186213        0.3389672433
N                -0.8030231164        0.6942244891        0.1202518820
C                -0.5701522073        2.0758402744        0.1422172196
C                 0.5807067482        2.6045860190        0.7203231915
C                 0.7807274045        3.9737639866        0.7174798080
C                -0.1644822613        4.8182095861        0.1600062296
C                -1.3258226502        4.2916620644       -0.3828693886
C                -1.5318139074        2.9255449961       -0.3955280586
H                 3.7724646117        0.9560145114        0.7149270447
H                 5.3565338311        0.5732776838        0.0387396747
H                 4.5165858830       -0.6122702531        1.0414800575
H                 3.5661826949        0.3579017468       -1.6987810070
H                 4.1606681774       -1.9080271071       -2.4816739735
H                 5.5948257471       -1.1328777960       -1.7952730593
H                 4.7040757393       -2.3502726642       -0.8668333684
H                 1.6234554633        0.8852741863       -0.6928444117
H                -0.2234890103       -2.4825010897       -0.1403587209
H                -4.4522671397       -0.2995034941        0.7965357231
H                -6.4283203805       -1.8032522441        1.1457096255
H                -4.1728208142       -5.1515274172        0.6167523492
H                -2.0995895747       -3.8340213235        0.2431469826
H                 1.2933853319        1.9556621190        1.2083331871
H                 1.6741800877        4.3820971068        1.1659173834
H                -0.0023618510        5.8851719508        0.1621987156
H                -2.0728966622        4.9488065468       -0.8013108761
H                -2.4310793916        2.4962138647       -0.8076846255


