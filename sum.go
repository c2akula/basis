package go_nd

import "github.com/c2akula/go.nd/nd"

// Sum computes the sum of the elements referenced by the iterator, x.
func Sum(x nd.Iterator) (s float64) {
	xd := x.Data()
	xi := x.Ind()
	for _, k := range xi {
		s += xd[k]
	}
	return
}

// Mean computes the average of the elements referenced by the iterator, x.
func Mean(x nd.Iterator) (m float64) {
	return Sum(x) * (1.0 / float64(x.Len()))
}
