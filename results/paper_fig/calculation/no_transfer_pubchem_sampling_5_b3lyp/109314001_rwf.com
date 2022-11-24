%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/109314001_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/109314001_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.7815531136        0.1188401629        1.4522272558
C                 6.3120902439        0.0405015533        1.1105277026
C                 5.8636457845       -0.6021816463       -0.0480496440
C                 4.5139900911       -0.6622326369       -0.3960308686
C                 3.5583029686       -0.0626933661        0.4367583844
N                 2.1714797770       -0.0600813518        0.1950087742
C                 1.4068759070       -0.5710374836       -0.8197888257
N                 1.9816346396       -1.2262397194       -1.8432919914
C                 1.1523808998       -1.6887572611       -2.7832527380
C                -0.2278537848       -1.5206838155       -2.7396696668
C                -0.7166043415       -0.8229312323       -1.6341830441
C                -2.2126366544       -0.6013102472       -1.5003668270
O                -2.9909574895       -1.0611657801       -2.3274308849
N                -2.5525505388        0.1356621933       -0.4034261518
C                -3.8737281616        0.4191386841        0.0487648585
C                -4.2216957390        0.0113123384        1.3528962002
C                -3.2354214163       -0.7346086180        2.2210517000
C                -5.5029545715        0.3056513735        1.8257039740
C                -6.4212226030        0.9818534349        1.0246255144
C                -6.0549113603        1.3868486233       -0.2557306775
C                -4.7767004092        1.1284542359       -0.7667384790
C                -4.3938737544        1.6293410907       -2.1360799378
N                 0.0795473341       -0.3481266859       -0.6747397156
C                 3.9930989189        0.5833736797        1.6069748047
C                 5.3429584315        0.6304166664        1.9334094088
H                 8.3492777555       -0.6771451504        0.9590184762
H                 8.2186918591        1.0753203449        1.1345921130
H                 7.9490285173        0.0315193518        2.5317942884
H                 6.5883375199       -1.0763858237       -0.7065432131
H                 4.1965090250       -1.1673645054       -1.2968297220
H                 1.6099202885        0.4108352933        0.8920782395
H                 1.6244302601       -2.2189694288       -3.6088016738
H                -0.9008375230       -1.8972208074       -3.4984836893
H                -1.7728516337        0.3263795990        0.2185012363
H                -2.8186968195       -1.6062516162        1.7030989407
H                -3.7152426232       -1.0827815793        3.1403107942
H                -2.3853858710       -0.1042668752        2.5193081321
H                -5.7822919915       -0.0097865906        2.8279798777
H                -7.4172452976        1.1988719586        1.4010451597
H                -6.7643859627        1.9298966557       -0.8751841235
H                -4.3391552509        0.8051708552       -2.8546074532
H                -3.4059866394        2.1050094625       -2.1219695620
H                -5.1213583836        2.3638019857       -2.4948907918
H                 3.2633397170        1.0487331391        2.2672767389
H                 5.6474547684        1.1352895390        2.8475911068


