package nd

// y += ax
func iaxpy(n int, a float64, x, y []float64, step int) {
	for i := 0; n != 0; i += step {
		y[i] += a * x[i]
		n--
	}
}

func ixpy(n int, x, y []float64, step int) {
	for i := 0; n != 0; i += step {
		y[i] += x[i]
		n--
	}
}

func uaxpy(n int, a float64, x, y []float64) {
	for i := range y[:n] {
		y[i] += a * x[i]
	}
}

func uxpy(n int, x, y []float64) {
	for i := range y[:n] {
		y[i] += x[i]
	}
}

// y = ax + y
func axpy2d(shape, strides Shape, a float64, x, y []float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		switch {
		case a == 0:
			return
		case a == 1:
			for i := 0; i < shape[0]; i++ {
				b := step0 * i
				ixpy(n, x[b:], y[b:], step1)
			}
		case a == -1:
			for i := 0; i < shape[0]; i++ {
				b := step0 * i
				iaxpy(n, -1, x[b:], y[b:], step1)
			}
		default:
			for i := 0; i < shape[0]; i++ {
				b := step0 * i
				iaxpy(n, a, x[b:], y[b:], step1)
			}
		}
		return
	}

	switch {
	case a == 0:
		return
	case a == 1:
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			uxpy(n, x[b:], y[b:])
		}
	case a == -1:
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			uaxpy(n, -1, x[b:], y[b:])
		}
	default:
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			uaxpy(n, a, x[b:], y[b:])
		}
	}
}

// Axpy computes y += a*x
func Axpy(a float64, x, y Array) Array {
	if !isShapeSame(x, y) {
		panic("Axpy: input arrays must have same shape")
	}
	ndims := y.Ndims()
	yshape := y.Shape()
	ystrides := y.Strides()
	xd := x.Data()
	yd := y.Data()

	if ndims < 3 {
		axpy2d(yshape, ystrides, a, xd, yd)
		return y
	}

	shape := make(Shape, ndims)
	copy(shape, yshape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}
	ind := make(Index, ndims-2)
	shape2d := yshape[ndims-2:]
	strides2d := ystrides[ndims-2:]
	istrides := ComputeStrides(shape)
	for i := 0; i < computeSize(shape); i++ {
		b := sub2ind(ystrides[:ndims-2], ind2sub(istrides[:ndims-2], i, ind))
		axpy2d(shape2d, strides2d, a, xd[b:], yd[b:])
	}

	return y
}
