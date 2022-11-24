%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/22751383.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                 8.0482203132        0.3112896131       -0.9587254863
C                 7.5651507888        0.3765500991        0.1447770757
O                 8.3155445646        0.2504003823        1.2533067127
C                 6.1320296301        0.5963842118        0.4258671982
C                 5.6428373487        0.6721431311        1.7272023918
C                 4.2951475746        0.8824450597        1.9430649134
C                 3.4157020637        1.0242963794        0.8753473451
C                 1.9477303128        1.2171684998        1.1101443860
C                 1.2139615818       -0.1312433489        1.0070047232
N                -0.2106043538        0.0172260948        1.1822775006
C                -1.0626203033        0.0671605130        0.1292510836
O                -0.7005797685       -0.0585991822       -1.0165090561
C                -2.5264805481        0.2337995404        0.5474254541
C                -3.1316665681       -1.1332353908        0.7591147738
C                -2.7744610681       -2.1115809602        1.6619987923
C                -3.4746676803       -3.3108245371        1.6557990949
C                -4.5084160588       -3.5244847164        0.7572965020
C                -4.8688616971       -2.5465720639       -0.1554257984
C                -4.1757487369       -1.3473835765       -0.1539676492
C                -4.3272162197       -0.1558619154       -0.9726565726
C                -5.2092696602        0.1372401762       -1.9995086648
C                -5.1249864967        1.3773409749       -2.6111481529
C                -4.1788240919        2.3069664561       -2.2073488729
C                -3.2921804798        2.0188128305       -1.1785248374
C                -3.3749853893        0.7887807573       -0.5638491354
C                 3.9101944503        0.9424874450       -0.4230672675
C                 5.2543980282        0.7322647768       -0.6488981053
H                 9.2413937124        0.1089680371        1.0020302342
H                 6.3225424654        0.5669950219        2.5581401685
H                 3.9197337164        0.9438041851        2.9546762054
H                 1.5319744238        1.8893457773        0.3574399433
H                 1.7750520453        1.6485234464        2.0988562995
H                 1.3683041785       -0.5553012774        0.0125148068
H                 1.6012442437       -0.8265740483        1.7574483903
H                -0.5499400443        0.2039862247        2.1120204580
H                -2.5807803389        0.8501706830        1.4503751525
H                -1.9662955034       -1.9585875455        2.3632771108
H                -3.2080057844       -4.0866293800        2.3583698620
H                -5.0370569603       -4.4663832120        0.7692665618
H                -5.6728363350       -2.7170615285       -0.8554868234
H                -5.9444455383       -0.5858520181       -2.3198646153
H                -5.8025597762        1.6233463577       -3.4158190519
H                -4.1284409381        3.2660068434       -2.7016142947
H                -2.5516702763        2.7436387414       -0.8728623966
H                 3.2314449586        1.0480516945       -1.2563721070
H                 5.6509932147        0.6705797475       -1.6508032528


