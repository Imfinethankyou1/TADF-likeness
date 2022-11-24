%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/102108738_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/102108738_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.8827439061       -1.1160459938        1.5886060040
C                -1.9307695021       -0.5334455318        0.1431951165
C                -1.2006038852       -1.5376777632       -0.7724263694
C                 0.2903296424       -1.6366551157       -0.3773619183
C                 1.0275487204       -0.2738125407       -0.3887012769
C                 0.2578584722        0.8296890398        0.3962465922
C                -1.2028865927        0.8323957840       -0.0860192974
C                -2.1829243577        1.9295448902        0.3181917997
C                -3.0254229347        1.8044882315        1.5573885108
C                -3.5649617458        1.3069212209        0.2336582367
C                -3.4146195895       -0.1528197418       -0.2104637730
C                -3.8498674589       -0.3034416161       -1.6919670940
C                -5.3514317681       -0.5794356069       -1.5817189566
C                -5.4761415420       -1.2652673758       -0.2266532181
O                -6.3914656667       -1.9184495370        0.2019564464
O                -4.3454035584       -1.0284123626        0.4959700240
C                 0.9195833499        2.1970356243        0.1992226533
C                 1.8491992717        2.7367783435        1.2437478795
C                 2.4099879391        2.2432827703       -0.0890651023
C                 3.1991125507        0.9950477524       -0.2342083081
C                 4.4912228097        1.0991689564       -0.6322233935
C                 5.3856033098       -0.0420462926       -0.7730904704
O                 6.6094037123        0.0624937047       -0.8125567370
C                 4.7685482331       -1.4430346395       -0.8133985817
O                 5.2420316544       -2.1742529516        0.3193542846
C                 3.2417825410       -1.4384585878       -0.8420766355
C                 2.5401415736       -0.3668262842        0.0356554886
C                 2.7122341576       -0.7452156515        1.5315683192
H                -0.9448358678       -0.8921077013        2.1013018140
H                -1.9765283156       -2.2059359397        1.5408813074
H                -2.7030640309       -0.7761537737        2.2169133057
H                -1.6600817204       -2.5328919748       -0.7124900972
H                -1.2610228967       -1.2193253933       -1.8208663283
H                 0.7794000746       -2.3203090720       -1.0784331709
H                 0.3730885842       -2.1053542886        0.6094239200
H                 1.0356552584        0.0653438999       -1.4382518147
H                 0.2736626667        0.5925362129        1.4651930644
H                -1.1378528753        0.9384030358       -1.1826037959
H                -2.0126864544        2.9266344880       -0.0806367804
H                -2.7535001910        1.1260490031        2.3541451749
H                -3.4781852676        2.7257985366        1.9152748328
H                -4.3975903669        1.8590964949       -0.1965369249
H                -3.6027044138        0.5816387930       -2.2843404534
H                -3.3556606474       -1.1648449327       -2.1488574641
H                -5.9501510917        0.3397031952       -1.5580914367
H                -5.7610893063       -1.2158858453       -2.3693273345
H                 0.3262534706        2.9304728379       -0.3422186307
H                 1.8910129949        3.8092217665        1.4129272615
H                 2.0263900565        2.1408593742        2.1335561158
H                 2.7573304009        3.0445371342       -0.7375949382
H                 4.9458423762        2.0773089496       -0.7766283151
H                 5.1361629076       -1.9215873563       -1.7361157903
H                 6.1994379109       -1.9994066087        0.3540625087
H                 2.9356976388       -1.2792495633       -1.8846948033
H                 2.9044490663       -2.4403211613       -0.5584702643
H                 2.2775995938       -0.0047114691        2.2081853782
H                 3.7719250845       -0.8518950106        1.7712003750
H                 2.2357029307       -1.7091703563        1.7392490614


