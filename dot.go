package go_nd

import (
	"github.com/c2akula/go.nd/nd/iter"
)

// Dot computes the inner product of two arrays, x and y.
// Note: x and y must have the same shape.
func Dot(x, y iter.Iterator) (s float64) {
	if x == y {
		xd := x.Data()
		for _, k := range x.Ind() {
			v := xd[k]
			s += v * v
		}
		return
	}

	xd := x.Data()
	yd := y.Data()
	yi := y.Ind()

	for i, l := range x.Ind() {
		k := yi[i]
		s += xd[l] * yd[k]
	}

	return
}
