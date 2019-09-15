package nd

import (
	"fmt"
	"strings"
)

func istr(n int, x []float64, str int, sb *strings.Builder) {
	sb.WriteByte('[')
	for j := 0; n != 0; j += str {
		sb.WriteString(fmt.Sprintf(" %8.4f ", x[j]))
	}
	sb.WriteString("]\n")
}

func ustr(n int, x []float64, sb *strings.Builder) {
	sb.WriteByte('[')
	for _, v := range x[:n] {
		sb.WriteString(fmt.Sprintf(" %8.4f ", v))
	}
	sb.WriteString("]\n")
}

func tostr(shp, str Shape, x []float64, sb *strings.Builder) {
	rn, cn := shp[0], shp[1]
	rs, cs := str[0], str[1]
	if cs != 1 {
		for i := 0; i < rn; i++ {
			b := i * rs
			istr(cn, x[b:], cs, sb)
		}
		return
	}

	for i := 0; i < rn; i++ {
		b := i * rs
		ustr(cn, x[b:], sb)
	}
}

func ndStr(x *Ndarray) string {
	var sb strings.Builder

	if x.ndims < 3 {
		sb.WriteByte('\n')
		sb.WriteString(fmt.Sprintf("shp: %v, str: %v\n", x.shape, x.strides))
		tostr(x.shape, x.strides, x.data, &sb)
		return sb.String()
	}

	ishp := make(Shape, x.ndims)
	copy(ishp, x.shape)
	for i := x.ndims - 2; i < x.ndims; i++ {
		ishp[i] = 1
	}
	istr := ComputeStrides(ishp)
	sub := make(Index, x.ndims)

	sb.WriteByte('\n')
	sb.WriteString(fmt.Sprintf("shp: %v, str: %v\n", x.shape, x.strides))
	for k := 0; k < ComputeSize(x.shape[:x.ndims-2]); k++ {
		Ind2sub(istr, k, sub)
		// print header
		sb.WriteByte('[')
		for _, i := range sub[:x.ndims-2] {
			sb.WriteString(fmt.Sprintf("%d,", i))
		}
		sb.WriteString(":,:]\n")
		b := Sub2ind(x.strides, sub)
		tostr(x.shape[x.ndims-2:], x.strides[x.ndims-2:], x.data[b:], &sb)
	}

	return sb.String()
}

func (x *Ndarray) String() string {
	return ndStr(x)
}
