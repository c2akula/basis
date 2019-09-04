package basis

import (
	"math"

	"github.com/c2akula/basis/nd"
)

func Dist(x, y nd.Iterator) (s float64) {
	if x.Len() != y.Len() {
		panic("Dist: input iterators must have the same size")
	}

	if x == y {
		return 0
	}

	xd := x.Data()
	xi := x.Ind()
	yd := y.Data()
	yi := y.Ind()

	for i, k := range xi {
		d := xd[k] - yd[yi[i]]
		s += d * d
	}
	return math.Sqrt(s)
}
