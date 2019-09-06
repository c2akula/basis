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
		n--
	}
	sb.WriteByte(']')
	return sb.String()
}

func uprint(n int, x []float64) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for _, v := range x[:n] {
		sb.WriteString(fmt.Sprintf(" %8.4f ", v))
	}
	sb.WriteByte(']')
	return sb.String()
}

func print2d(shape, strides Shape, x []float64) string {
	var sb strings.Builder
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 != 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			sb.WriteString(iprint(n, x[b:], step1))
			sb.WriteByte('\n')
		}
		return sb.String()
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		sb.WriteString(uprint(n, x[b:]))
		sb.WriteByte('\n')
	}
	return sb.String()
}

func (array *ndarray) String() string {
	ndims := array.ndims

	if ndims < 3 {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("shp: %v, str: %v\n", array.shape, array.strides))
		sb.WriteString(print2d(array.shape, array.strides, array.data))
		return sb.String()
	}

	ishp := make(Shape, ndims)
	copy(ishp, array.shape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		ishp[i] = 1
	}

	shp2d := array.shape[ndims-2:]
	str2d := array.strides[ndims-2:]
	shpnd := array.shape[:ndims-2]

	isub := make(Index, ndims)
	istr := ComputeStrides(shpnd)
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("shp: %v, str: %v\n", array.shape, array.strides))
	for k := 0; k < ComputeSize(shpnd); k++ {
		sb.WriteByte('[')
		for _, k := range Ind2sub(istr, k, isub[:ndims-2]) {
			sb.WriteString(fmt.Sprintf("%d,", k))
		}
		sb.WriteString(":,:]\n")
		sb.WriteString(print2d(shp2d, str2d, array.data[array.sub2ind(isub):]))
	}

	return sb.String()
}
