package go_nd

import (
	"github.com/c2akula/go.nd/nd"
)

func scale(a float64, x, y nd.Iterator) {
	if x.Len() != y.Len() {
		panic("input iterators must have same size.")
	}

	xd := x.Data()
	if x == y {
		for _, k := range x.Ind() {
			xd[k] *= a
		}
		return
	}

	yd := y.Data()
	xi := x.Ind()
	yi := y.Ind()
	for i, k := range xi {
		yd[yi[i]] = a * xd[k]
	}
}

func iscale(n int, a float64, x, y []float64, step int) {
	for i := 0; n != 0; i += step {
		y[i] = a * x[i]
		n--
	}
}

func uscale(n int, a float64, x, y []float64) {
	for i, v := range x[:n] {
		y[i] = a * v
	}
}

func scale2d(shp, str nd.Shape, a float64, x, y []float64) {
	n := shp[1]
	step0, step1 := str[0], str[1]
	if step1 > 1 {
		for i := 0; i < shp[0]; i++ {
			b := step0 * i
			iscale(n, a, x[b:], y[b:], step1)
		}
		return
	}

	for i := 0; i < shp[0]; i++ {
		b := step0 * i
		uscale(n, a, x[b:], y[b:])
	}
}

// Scale performs the element-wise operation y = a*x.
func Scale(a float64, x, y nd.Array) nd.Array {
	if !nd.IsShapeSame(x, y) {
		panic("input arrays must have the same shape")
	}

	ndims := x.Ndims()
	xshp, xstr := x.Shape(), x.Strides()
	xd := x.Data()
	yd := y.Data()
	if ndims < 3 {
		scale2d(xshp, xstr, a, xd, yd)
		return y
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
		scale2d(shp2d, str2d, a, xd[b:], yd[b:])
	}

	return y
}
