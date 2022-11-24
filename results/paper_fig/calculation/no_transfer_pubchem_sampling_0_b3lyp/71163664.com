%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/71163664.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.4802591593       -1.9193703053       -0.8211106953
C                 4.6464208667       -0.4472909986       -0.4187178805
N                 3.7274503648       -0.0805382139        0.6324596973
C                 3.9205966599       -0.4133485482        1.9222596687
O                 4.8445063345       -1.0943297892        2.3099614841
C                 2.8596934956        0.1466522130        2.8743666654
N                 1.5666601738        0.3164056312        2.2652419617
C                 1.0603172645        1.5260896146        1.9613487736
O                 1.6120967272        2.5761542808        2.2089371545
C                -0.2989379227        1.4831308911        1.2640610741
O                -0.6837020574        0.1526777666        1.0101065215
C                -1.8898113493       -0.0729810343        0.4018470622
C                -2.1971424730       -1.4371765976        0.1952896518
C                -3.3680552113       -1.8003680909       -0.3977827668
C                -4.3004560502       -0.8275730595       -0.8194033481
C                -5.5206625437       -1.1704479436       -1.4343173251
C                -6.4000406650       -0.2008451952       -1.8299927965
C                -6.0952913934        1.1534141843       -1.6272598357
C                -4.9187457442        1.5169455819       -1.0330959629
C                -3.9914126440        0.5411688103       -0.6135438837
C                -2.7722348691        0.8965099849        0.0027747805
C                 4.3627137074        0.4394373802       -1.6157955478
O                 3.3844146369        1.1299724643       -1.7250359969
O                 5.3056279893        0.3480777043       -2.5487192835
H                 5.1797276444       -2.1693750704       -1.6133011797
H                 4.6843304792       -2.5407089920        0.0463080757
H                 3.4644570509       -2.1043607345       -1.1633398151
H                 5.6777005193       -0.2912431278       -0.0795153000
H                 2.9851539779        0.5517958967        0.3630395299
H                 3.1918680221        1.1371845646        3.2045731588
H                 2.7985933418       -0.5118821604        3.7447127718
H                 1.0377267356       -0.5010443969        2.0010765375
H                -1.0252559128        1.9909101873        1.9133829186
H                -0.2121549890        2.0578655644        0.3319392770
H                -1.4783493010       -2.1729064913        0.5215345730
H                -3.5983401154       -2.8444150451       -0.5528280473
H                -5.7484925019       -2.2158216004       -1.5871716334
H                -7.3341672722       -0.4696848186       -2.3007232266
H                -6.7997588959        1.9080235683       -1.9452193390
H                -4.6829428289        2.5599789128       -0.8763929988
H                -2.5648780091        1.9458042203        0.1447587138
H                 5.0905195981        0.9075117916       -3.3127841891


