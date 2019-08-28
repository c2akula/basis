package go_nd

import (
	"github.com/c2akula/go.nd/nd"
)

// Dot computes the inner product of elements referenced by the iterators, x and y.
// Note: x and y must have the same shape.
func Dot(x, y nd.Iterator) (s float64) {
	if x.Len() != y.Len() {
		panic("iterators must have same length")
	}
	xd, xi := x.Iter()
	if x == y {
		for _, k := range xi {
			v := xd[k]
			s += v * v
		}
		return
	}

	yd, yi := y.Iter()
	for i, l := range xi {
		k := yi[i]
		s += xd[l] * yd[k]
	}

	return
}

func idot(n int, x, y []float64, step int) (s float64) {
	for i := 0; n != 0; i += step {
		s += x[i] * y[i]
		n--
	}
	return
}

func udot(n int, x, y []float64) (s float64) {
	for i, v := range x[:n] {
		s += v * y[i]
	}
	return
}

func dot2d(shape, strides nd.Shape, x, y []float64) (s float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			s += idot(n, x[b:], y[b:], step1)
		}
	}
	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		s += udot(n, x[b:], y[b:])
	}

	return
}

func dot(x, y nd.Array) (s float64) {
	if !nd.IsShapeSame(x, y) {
		panic("arrays must have same shape")
	}

	ndims := x.Ndims()
	xshp, xstr := x.Shape(), x.Strides()
	xd := x.Data()
	yd := y.Data()
	if ndims < 3 {
		return dot2d(xshp, xstr, xd, yd)
	}

	shp := make(nd.Shape, ndims)
	copy(shp, xshp[:ndims-2])
	for k := ndims - 2; k < ndims; k++ {
		shp[k] = 1
	}
	shp2d := xshp[ndims-2:]
	str2d := xstr[ndims-2:]
	istr := nd.ComputeStrides(shp)
	ind := make(nd.Index, ndims-2)
	for k := 0; k < nd.ComputeSize(shp); k++ {
		b := nd.Sub2ind(xstr[:ndims-2], nd.Ind2sub(istr[:ndims-2], k, ind))
		s += dot2d(shp2d, str2d, xd[b:], yd[b:])
	}
	return
}
