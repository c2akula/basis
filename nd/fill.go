package nd

func ifill(n int, v float64, x []float64, step int) {
	for i := 0; n != 0; i += step {
		x[i] = v
		n--
	}
}

func ufill(n int, v float64, x []float64) {
	for i := range x[:n] {
		x[i] = v
	}
}

func fill2d(shape, strides Shape, v float64, x []float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			ifill(n, v, x[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		ufill(n, v, x[b:])
	}
}

func Fill(x Array, v float64) Array {
	ndims := x.Ndims()
	xshape := x.Shape()
	xstrides := x.Strides()
	xd := x.Data()

	if ndims < 3 {
		fill2d(xshape, xstrides, v, xd)
		return x
	}

	shape := make(Shape, ndims)
	copy(shape, xshape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims-2)
	shape2d := xshape[ndims-2:]
	strides2d := xstrides[ndims-2:]
	istrides := ComputeStrides(shape)
	for i := 0; i < computeSize(shape); i++ {
		b := sub2ind(xstrides[:ndims-2], ind2sub(istrides[:ndims-2], i, ind))
		fill2d(shape2d, strides2d, v, xd[b:])
	}
	return x
}
