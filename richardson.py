#!/usr/bin/env python

"""

	Richardson Extrapolation


	Approximation of second derivative...

	f( x + h )    =    f(x)  +  (h) * f'(x)  
		+  (h^2/2) * f''(x)    +    (h^3/3!) * f'''(x)  +  ...        eq (1)

	f( x - h )    =    f(x)  -  (h) * f'(x)  
		+  (h^2/2) * f''(x)    -    (h^3/3!) * f'''(x)  +  ...        eq (2)


	Add eq's (1) and (2)...

	f( x + h ) + f( x - h )   =   2 * f(x)  
		+  (h^2) * f''(x)    +    2 * [ (h^4/4!) * f^(4)(x)  +  (h^6/6!) * f^(6)(x)  +  ... ]


	Rearrange:


	f''(x)   =  (1/h^2) * [ f( x + h ) 
		 -  2 * f(x)  +  f( x - h ) ]    -2 * [ (h^4/4!) * f^(4)(x)  +  (h^6/6!) * f^(6)(x)  +  ... ]


	Let,
		 	
	r4( h )   =   (1/h^2) * [ f( x + h )  -  2 * f(x)  +  f( x - h ) ]


	Then,

	f''(x)   =   r4( h )  +  O(h^4)


	Continuing...


	r6( h )   =   -1/(2^2 - 1) * r4( h )   +   (2^2)/(2^2 - 1) * r4( h/2 )

	r8( h )   =   -1/(2^4 - 1) * r6( h )   +   (2^4)/(2^4 - 1) * r6( h/2 )

	r10( h )  =   -1/(2^6 - 1) * r6( h )   +   (2^6)/(2^6 - 1) * r6( h/2 )

	.
	.
	.

	rn( h )  =   -1/(2^(n-1) - 1) * rn-1( h )   +   (2^(n-1))/(2^(n-1) - 1) * rn-1( h/2 )


	error in rn( h ) apprixmation of f''(x) is  O(h^n)

"""

__author__  = "LJ Brown"
__file__ = "richardson.py"

import math
import numpy as np

def richardson(f, x, h, order=10):
	""" second derivative approximation using richardson extrapolation """

	# O(h^4)
	init_order = 4

	ddf_approx = lambda h0: ( 1.0 / h0**2 ) * ( f( x + h0 )  -  2 * f(x)  +  f( x - h0 ) )
	get_ddf_approx = lambda prev_ddf_approx, o: \
		lambda hi: -1.0/(2.0**(o-2) - 1.0) * prev_ddf_approx(hi) + (2.0**(o-2))/(2.0**(o-2) - 1.0) * prev_ddf_approx(hi/2) 

	assert order >= init_order

	if order == init_order:
		return ddf_approx(h)

	ddf_approx = ddf_approx
	for i in range(init_order+2, order+2, 2):
		ddf_approx = get_ddf_approx(ddf_approx, i)


	return ddf_approx(h)
