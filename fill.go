package go_nd

import (
	"github.com/c2akula/go.nd/nd"
)

func fill(a float64, x nd.Iterator) {
	xd, xi := x.Iter()
	for _, k := range xi {
		xd[k] = a
	}
}

func ifill(n int, a float64, x []float64, step int) {
	for i := 0; n != 0; i += step {
		x[i] = a
		n--
	}
}

func ufill(n int, a float64, x []float64) {
	for i := range x[:n] {
		x[i] = a
	}
}

func fill2d(shp, str nd.Shape, a float64, x []float64) {
	n := shp[1]
	step0, step1 := str[0], str[1]
	if step1 > 1 {
		for i := 0; i < shp[0]*shp[1]; i += step0 {
			ifill(n, a, x[i:], step1)
		}
		return
	}

	for i := 0; i < shp[0]*shp[1]; i += step0 {
		ufill(n, a, x[i:])
	}
}

func Fill(a float64, x nd.Array) nd.Array {
	ndims := x.Ndims()
	xshp, xstr := x.Shape(), x.Strides()
	xd := x.Data()

	if x.Size() < _nel {
		fill(a, x.Range())
		return x
	}

	if ndims < 3 {
		fill2d(xshp, xstr, a, xd)
		return x
	}

	shp2d := xshp[ndims-2:]
	str2d := xstr[ndims-2:]
	shpnd := xshp[:ndims-2]
	strnd := xstr[:ndims-2]
	step := strnd[ndims-3]
	for k := 0; k < nd.ComputeSize(shpnd); k++ {
		fill2d(shp2d, str2d, a, xd[k*step:])
	}

	return x
}
