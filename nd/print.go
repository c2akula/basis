package nd

import (
	"fmt"
	"strconv"
	"strings"
)

func print2d(array Array) string {
	var sb strings.Builder
	it := array.Take()
	if it == nil {
		it = Iter(array)
	}
	nc := array.Shape()[1]

	for i := 0; i < array.Size(); i += nc {
		sb.WriteByte('[')
		for row := it.From(i).To(i + nc - 1); !row.Done(); row.Next() {
			sb.WriteString(fmt.Sprintf(" %8.4f ", *row.Upk()))
		}
		sb.WriteString("]\n")
	}
	it.Reset()
	return sb.String()
}

func (array *Ndarray) String() string {
	if array.Ndims() < 3 {
		return print2d(array)
	}

	var sb strings.Builder
	ndims := array.Ndims()
	shape := array.Shape()

	ind := make(Index, array.Ndims())
	end := make(Index, array.Ndims())
	computeEnd(array.Shape(), end)
	for it := Subs(ind[:ndims-2], end[:ndims-2]); !it.Done(); it.Next() {
		copy(ind[:ndims-2], it.I())
		sb.WriteByte('[')
		for _, i := range ind[:ndims-2] {
			sb.WriteString(strconv.Itoa(i) + ",")
		}
		sb.WriteString(":,:]\n")
		sb.WriteString(print2d(array.View(ind, shape[ndims-2:])))
	}
	return sb.String()
}
