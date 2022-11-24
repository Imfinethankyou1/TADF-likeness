%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/76311879_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/76311879_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.1620385446        1.2033197280        1.3519359206
O                 6.1706799627        0.6550438923        0.4981783888
C                 4.9354484020        0.4031670293        1.0243362879
C                 4.5685516328        0.6468805581        2.3524795029
C                 3.2655830914        0.3384681113        2.7520166285
C                 2.3261817021       -0.1997885253        1.8783443222
C                 2.7073399486       -0.4391637798        0.5468239732
N                 1.8378832525       -0.9753580269       -0.4257101740
C                 0.5292419961       -1.3778129617       -0.2817268338
O                -0.1186168889       -1.3607136509        0.7621689964
N                 0.0038841654       -1.9048042478       -1.4604935999
N                -1.3816697392       -2.1375093579       -1.4279905678
C                -2.0934238705       -1.7458784443       -2.5307371092
O                -1.5253825288       -1.3962306988       -3.5640695334
C                -3.5968856560       -1.8068153259       -2.3509999383
C                -4.2324038618       -0.3922229930       -2.3129860982
C                -3.5148269217        0.6494458490       -1.4741211575
C                -3.1062885119        1.8508088243       -2.0593115298
C                -2.4689673461        2.8547970563       -1.3266207207
C                -2.2236789610        2.6544034983        0.0280053501
C                -2.6096438236        1.4618832927        0.6446816777
C                -3.2518692519        0.4674439774       -0.0990744545
O                -3.6525178532       -0.7346111813        0.4347537022
C                -3.4455760257       -0.9568295489        1.8279408257
C                 4.0064001811       -0.1361471910        0.1287846157
H                 6.8604212447        2.1834931650        1.7443006970
H                 8.0552408902        1.3213470544        0.7356104168
H                 7.3874426187        0.5332917953        2.1921097355
H                 5.2700855380        1.0640985122        3.0648174861
H                 2.9752997783        0.5236975564        3.7829764186
H                 1.3245573258       -0.4385534252        2.2037167772
H                 2.2624325757       -1.1553942669       -1.3260644753
H                 0.2254759136       -1.4695501076       -2.3571678272
H                -1.7693547155       -2.0417120591       -0.4935752755
H                -4.0145546609       -2.3570581742       -3.2010374649
H                -3.8436907665       -2.3553235805       -1.4398873387
H                -4.2877622165       -0.0125585950       -3.3378780766
H                -5.2657568434       -0.5098403016       -1.9608435701
H                -3.2943988445        1.9981038785       -3.1200539504
H                -2.1662134346        3.7771310751       -1.8136468427
H                -1.7233543497        3.4185584028        0.6167688306
H                -2.3951220909        1.3116517190        1.6957570089
H                -3.8692871732       -1.9415374799        2.0352093173
H                -2.3775476248       -0.9579285679        2.0709783680
H                -3.9687635627       -0.2041989859        2.4309749349
H                 4.3233667591       -0.3094954985       -0.8963346443


