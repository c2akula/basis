package nd

func iscale(n int, a float64, x []float64, step int) {
	for i := 0; n != 0; i += step {
		x[i] *= a
		n--
	}
}

func uscale(n int, a float64, x []float64) {
	for i := range x[:n] {
		x[i] *= a
	}
}

func scale2d(shape, strides Shape, a float64, x []float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]

	if a == 0 {
		fill2d(shape, strides, a, x)
		return
	}

	if step1 > 1 {
		switch a {
		case 1:
			return
		default:
			for i := 0; i < shape[0]; i++ {
				b := step0 * i
				iscale(n, a, x[b:], step1)
			}
		}
		return
	}

	switch a {
	case 1:
		return
	default:
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			uscale(n, a, x[b:])
		}
	}

}

func Scale(x Array, a float64) Array {
	ndims := x.Ndims()
	xshape := x.Shape()
	xstrides := x.Strides()
	xd := x.Data()

	if ndims < 3 {
		scale2d(xshape, xstrides, a, xd)
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
		scale2d(shape2d, strides2d, a, xd[b:])
	}

	return x
}
