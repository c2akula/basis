package nd

import (
	"fmt"
	"strconv"
	"strings"
)

func print2d(array Array) string {
	var sb strings.Builder
	shape := array.Shape()
	ind := make(Index, 2)
	for i := 0; i < shape[0]; i++ {
		ind[0] = i
		sb.WriteByte('[')
		for j := 0; j < shape[1]; j++ {
			ind[1] = j
			sb.WriteString(fmt.Sprintf(" %8.4f ", array.Get(ind)))
		}
		sb.WriteString("]\n")
	}
	return sb.String()
}

func (array *Ndarray) String() string {
	if array.ndims < 3 {
		return print2d(array)
	}

	ndims := array.Ndims()
	shape := make(Shape, ndims)
	copy(shape, array.Shape()[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims)
	b := array.View(ind, shape)

	var sb strings.Builder
	shape = array.Shape()
	for it := Iter(b); !it.Done(); it.Next() {
		copy(ind[:ndims-2], it.I()[:ndims-2])
		sb.WriteByte('[')
		for _, i := range ind[:ndims-2] {
			sb.WriteString(strconv.Itoa(i) + ",")
		}
		sb.WriteString(":,:]\n")
		sb.WriteString(print2d(array.View(ind, shape[ndims-2:])))
	}
	return sb.String()
}
