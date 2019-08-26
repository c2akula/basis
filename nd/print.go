package nd

import (
	"fmt"
	"strings"
)

func iprint(n int, x []float64, step int) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for i := 0; n != 0; i += step {
		sb.WriteString(fmt.Sprintf(" %8.4f ", x[i]))
	}
	sb.WriteString("]\n")
	return sb.String()
}

func uprint(n int, x []float64) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for _, v := range x[:n] {
		sb.WriteString(fmt.Sprintf(" %8.4f ", v))
	}
	sb.WriteString("]\n")
	return sb.String()
}

func print2d(shape, strides Shape, x []float64) string {
	var sb strings.Builder
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			sb.WriteString(iprint(n, x[b:], step1))
		}
		return sb.String()
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		sb.WriteString(uprint(n, x[b:]))
	}
	return sb.String()
}

func (array *ndarray) String() string {
	var sb strings.Builder

	if array.ndims < 3 {
		return print2d(array.shape, array.strides, array.data)
	}

	ndims := array.ndims
	shape := make(Shape, ndims)
	copy(shape, array.shape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims-2)
	shape2d := array.shape[ndims-2:]
	strides2d := array.strides[ndims-2:]
	istrides := ComputeStrides(shape)

	for i := 0; i < ComputeSize(shape); i++ {
		b := Sub2ind(array.strides[:ndims-2], Ind2sub(istrides[:ndims-2], i, ind))
		sb.WriteString(print2d(shape2d, strides2d, array.data[b:]))
	}

	return sb.String()
}
