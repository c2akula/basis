package go_nd

import (
	"github.com/c2akula/go.nd/nd/iter"
)


// Axpy performs the operation y += a*x
func Axpy(a float64, x, y iter.Iterator) iter.Iterator {

	if x.Len() != y.Len() {
		panic("input iterators must have same size")
	}

	xd := x.Data()
	xi := x.Ind()
	// x = a*x + x
	if x == y {
		switch a {
		case 0:
			return x
		case 1:
			Scale(2, x, x)
		case -1:
			Fill(x, 0)
		default:
			for _, k := range xi {
				v := xd[k]
				xd[k] = a*v + v
			}
		}
		return x
	}

	yd := y.Data()
	yi := y.Ind()

	// y = a*x + y
	switch a {
	case 0:
		return y
	case 1:
		for i, k := range yi {
			l := xi[i]
			xv := xd[l]
			yv := yd[k]
			yd[k] = xv + yv
		}
	case -1:
		for i, k := range yi {
			l := xi[i]
			xv := xd[l]
			yv := yd[k]
			yd[k] = yv - xv
		}
	default:
		for i, k := range yi {
			l := xi[i]
			xv := xd[l]
			yv := yd[k]
			yd[k] = a*xv + yv
		}
	}

	return y
}
