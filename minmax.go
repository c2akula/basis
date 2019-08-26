package go_nd

import (
	"github.com/c2akula/go.nd/nd"
)

// Min returns the location of the smallest element in the array, x.
func Min(x nd.Array) (k int) {
	it := x.Take()
	if it == nil {
		it = nd.Iter(x)
	}

	xd := x.Data()
	ind := it.Indices()
	m := xd[ind[0]]
	k = ind[0]
	for _, j := range ind[1:] {
		if v := xd[j]; v < m {
			m = v
			k = j
		}
	}

	return
}

// Max returns the location of the largest element in the array, x.
func Max(x nd.Array) (k int) {
	it := x.Take()
	if it == nil {
		it = nd.Iter(x)
	}

	xd := x.Data()
	ind := it.Indices()
	m := xd[ind[0]]
	k = ind[0]
	for _, j := range ind[1:] {
		if v := xd[j]; v > m {
			m = v
			k = j
		}
	}

	return
}
