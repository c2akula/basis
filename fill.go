package go_nd

import "github.com/c2akula/go.nd/nd/iter"

func Fill(x iter.Iterator, v float64) {
	xd := x.Data()
	for _, k := range x.Ind() {
		xd[k] = v
	}
}
