%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/126729624.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -4.4199630584        1.0789474045        2.6942620427
C                -4.8109986261        0.7904045081        1.4462443935
C                -3.8997477083        0.6108886682        0.4472955964
C                -2.4201441138        0.7428204222        0.6556622079
N                -1.6905762273       -0.2145613700       -0.1389889856
C                -0.3119259483       -0.2134068520       -0.1978801370
C                 0.4816880285        0.4198736763        0.7591363377
C                 1.8582453433        0.3435434037        0.6868158173
C                 2.4948606323       -0.3663758400       -0.3276167218
C                 3.9598055383       -0.4437012865       -0.3952897861
C                 4.5942500237       -1.6342270398       -0.7451282865
C                 5.9731025689       -1.7071437325       -0.8076132817
C                 6.7438554481       -0.5920317936       -0.5246342207
C                 6.1251707535        0.5970560249       -0.1768865230
C                 4.7465293342        0.6710590913       -0.1113196773
C                 1.6976650085       -1.0013412472       -1.2795274274
C                 0.3242535884       -0.9215375660       -1.2215464757
O                -4.2850664505        0.3260806081       -0.8212616743
C                -5.5872680049        0.2104998560       -1.0958046668
C                -6.5632300257        0.3651551733       -0.1778673643
C                -6.2230530735        0.6686344355        1.1718197735
S                -7.3118089973        0.8783509723        2.4075667593
H                -5.2468022248        1.1405026659        3.2272015512
H                -2.1231643992        1.7804247567        0.4159740778
H                -2.2223221302        0.5776484594        1.7197964833
H                -2.1454989186       -0.4731073903       -1.0023106746
H                 0.0267490122        0.9705455880        1.5673175741
H                 2.4514145549        0.8290327376        1.4480353389
H                 3.9996246393       -2.5130467562       -0.9488108795
H                 6.4500934004       -2.6391563244       -1.0754640253
H                 7.8213334758       -0.6494325047       -0.5746339886
H                 6.7210287810        1.4717979229        0.0415208382
H                 4.2700525297        1.6068591216        0.1430400792
H                 2.1639404816       -1.5407156624       -2.0910565070
H                -0.2774169709       -1.4112628084       -1.9750540081
H                -5.7641171560       -0.0209990354       -2.1356045373
H                -7.6005571086        0.2619207127       -0.4474790223


