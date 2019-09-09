package nd

import (
	"fmt"
	"strings"
)

func (x *Ndarray) String() string {
	var sb strings.Builder

	it := newiterator(x, 2)
	sb.WriteByte('\n')
	sb.WriteString(fmt.Sprintf("shp: %v, str: %v\n", x.shape, x.strides))
	for v, ok := it.init(); ok; v, ok = it.next() {
		// print header
		sb.WriteByte('[')
		for _, i := range it.sub {
			sb.WriteString(fmt.Sprintf("%d,", i))
		}
		sb.WriteString(":,:]\n")

		for i := 0; i < it.dn[0]; i++ {
			k := i * it.ds[0]
			sb.WriteByte('[')
			for j := 0; j < it.dn[1]; j++ {
				sb.WriteString(fmt.Sprintf(" %8.4f ", v[k+j*it.ds[1]]))
			}
			sb.WriteString("]\n")
		}
	}
	return sb.String()
}
