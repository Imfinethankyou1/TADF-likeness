%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/156420564.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
F                 9.5651554950        1.7819390729       -0.1603661950
C                 9.2638310421        0.5441747367       -0.5919467075
F                10.2143127185       -0.2688618709       -0.0779734599
F                 9.5060271661        0.6046524775       -1.9142835894
C                 7.8612328210        0.1109447001       -0.2585891115
C                 7.2311431938        0.6152961139        0.8938465631
C                 7.9194558305        1.2819127506        1.9420031049
C                 7.2546084246        1.7721835366        3.0254149507
C                 5.8580285984        1.6387016745        3.1369136178
C                 5.1648191957        0.9668455894        2.1787419593
C                 5.8223409229        0.4124518802        1.0498995374
C                 5.1121543111       -0.3264677976        0.0968758876
C                 3.6418232416       -0.4616036900        0.2190773412
C                 3.0983229106       -1.5238515068        0.9479872525
C                 1.7321601693       -1.6706112247        1.0759779886
C                 0.9032647384       -0.7423176182        0.4662965577
S                -0.8457730368       -0.7476256822        0.5066323007
C                -0.8851765038        0.7026416118       -0.4723354277
C                -2.0436992647        1.3512872310       -0.8863392235
C                -3.3689746693        0.8190254762       -0.4992327634
C                -3.9696242097        1.2537059863        0.6879040386
C                -3.3514173890        2.2134570179        1.5312516180
C                -3.9474025727        2.6229885028        2.6837430135
C                -5.2000233319        2.0992734146        3.0635101456
C                -5.8255645267        1.1772123726        2.2825227750
C                -5.2380570799        0.7205838921        1.0736617630
C                -5.8671485877       -0.2323734370        0.2634244767
C                -7.1803929059       -0.7874349105        0.6659907801
C                -7.2472775963       -1.9319144582        1.4526884298
C                -8.4738343327       -2.4497875709        1.8283686392
C                -9.6440887595       -1.8305250819        1.4224548787
C                -9.5841084749       -0.6903974346        0.6387981578
C                -8.3588373018       -0.1703170533        0.2617342329
C                -5.2627852053       -0.6682620305       -0.9219506531
C                -5.8760295269       -1.6356358830       -1.7604780386
C                -5.2747958761       -2.0541029777       -2.9071403742
C                -4.0226522876       -1.5298625014       -3.2877356378
C                -3.4026625847       -0.5987198313       -2.5133277680
C                -3.9945278456       -0.1358304925       -1.3091826876
C                -1.9000047518        2.4927738802       -1.6604366470
C                -0.6404385381        2.9694272302       -2.0088065486
C                 0.5038130430        2.3197685404       -1.5938898303
C                 0.3924898230        1.1706673305       -0.8152818876
C                 1.4293030756        0.3328512811       -0.2710731794
C                 2.8089646930        0.4631298105       -0.3879791206
C                 5.7831675408       -0.9694096420       -0.9495072717
C                 5.0868609376       -1.8164287153       -1.8504051919
C                 5.7431518156       -2.4862122747       -2.8356899324
C                 7.1394851727       -2.3590670560       -2.9577233056
C                 7.8406006135       -1.5314913571       -2.1332980654
C                 7.1913145692       -0.7764892225       -1.1208387130
H                 8.9922664164        1.3709749873        1.9099311960
H                 7.8041233949        2.2609281991        3.8157621303
H                 5.3495909574        2.0516738857        3.9950417432
H                 4.0983353639        0.8242323570        2.2686244170
H                 3.7641742000       -2.2346887289        1.4141998927
H                 1.3125086467       -2.4899830687        1.6384024707
H                -2.3921642943        2.6152418047        1.2379047812
H                -3.4659059314        3.3533184997        3.3175330171
H                -5.6574870915        2.4378873675        3.9815765656
H                -6.7852197854        0.7735777382        2.5708802033
H                -6.3313780645       -2.4116255002        1.7669285193
H                -8.5165620642       -3.3394287280        2.4398110049
H               -10.6011874216       -2.2357014256        1.7162978731
H               -10.4948187472       -0.2043987351        0.3200809727
H                -8.3076622574        0.7192895197       -0.3497601158
H                -6.8356553778       -2.0350346461       -1.4661838616
H                -5.7516624214       -2.7915514090       -3.5360664155
H                -3.5609820815       -1.8748772465       -4.2013734467
H                -2.4441746963       -0.1940038566       -2.8051444521
H                -2.7862767403        3.0124963863       -1.9930183492
H                -0.5615272863        3.8609176037       -2.6127630033
H                 1.4790365767        2.6930270911       -1.8664979467
H                 3.2319706331        1.2821411100       -0.9500364412
H                 4.0210864220       -1.9318635967       -1.7215837303
H                 5.2054590939       -3.1330307693       -3.5124180381
H                 7.6594267448       -2.9340449534       -3.7090798856
H                 8.9121499074       -1.4877656757       -2.2332957805


