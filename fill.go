package go_nd

import "github.com/c2akula/go.nd/nd"

func Fill(x nd.Iterator, v float64) {
	xd := x.Data()
	for _, k := range x.Ind() {
		xd[k] = v
	}
}
