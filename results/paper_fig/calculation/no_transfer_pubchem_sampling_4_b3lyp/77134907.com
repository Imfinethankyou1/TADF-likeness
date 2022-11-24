%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/77134907.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 3.3060651678        2.9496404384       -0.0845754587
O                 4.2385843747        1.9374605472       -0.3626938108
C                 3.7893762432        0.6562326761       -0.4800972545
C                 2.4741832324        0.2595777517       -0.3195710487
C                 2.1022591960       -1.0804220835       -0.4627553429
C                 0.7204417033       -1.4949056605       -0.2890480792
N                 0.2411956076       -2.6395674061       -0.8344363007
C                -1.0328750296       -2.7086930654       -0.5043706375
C                -2.0092026889       -3.6637473811       -0.8233765854
C                -3.2793195289       -3.4985931161       -0.3524605734
C                -3.6256638426       -2.3851219634        0.4553985021
C                -5.0272877409       -2.2419385884        0.9605978881
C                -2.6680335213       -1.4639745324        0.7584171089
N                -1.4063928247       -1.6201150827        0.2846574699
C                -0.2883403723       -0.8438187286        0.4149163241
N                -0.2719992619        0.2869192801        1.2317972133
C                -0.8328374910        1.4992940289        0.8754715640
C                -0.5924922499        2.6227476144        1.6707133939
C                -1.1725843980        3.8374974487        1.3681697208
C                -1.9967872037        3.9289063330        0.2609594125
F                -2.5690290943        5.1186395602       -0.0390097796
C                -2.2437578780        2.8367069389       -0.5497296253
C                -1.6598438612        1.6235000885       -0.2430815070
C                 3.0678058296       -2.0229763236       -0.7899776719
C                 4.3857704884       -1.6355260389       -0.9476733293
C                 4.7641526434       -0.3126129554       -0.7903331728
O                 6.0526876257        0.0995296073       -0.9331763467
H                 3.8785301323        3.8752870083       -0.0561259987
H                 2.5395206885        3.0230504264       -0.8635542936
H                 2.8187386314        2.7981931637        0.8851909752
H                 1.7161810161        0.9982474770       -0.1185924932
H                -1.7233035165       -4.5023245970       -1.4355373771
H                -4.0455783223       -4.2207740186       -0.5890059354
H                -5.7263700206       -2.2033951303        0.1264567160
H                -5.1375552854       -1.3363151778        1.5508975623
H                -5.2945794004       -3.0979051932        1.5787801664
H                -2.8528626755       -0.5933745973        1.3697627847
H                 0.4966243099        0.3359235169        1.8858226563
H                 0.0547120361        2.5347181816        2.5315714444
H                -0.9911449329        4.7088315334        1.9769611223
H                -2.8823003715        2.9415500045       -1.4123197674
H                -1.8334840286        0.7713153161       -0.8817145337
H                 2.7708070109       -3.0506869066       -0.9174016050
H                 5.1395655177       -2.3704172934       -1.1983176943
H                 6.6064270858       -0.6565651008       -1.1633018027


