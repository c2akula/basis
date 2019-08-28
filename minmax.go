package go_nd

import "github.com/c2akula/go.nd/nd"

// Min returns the value and location of the smallest element in the elements referenced by iterator, x.
func Min(x nd.Iterator) (m float64, k int) {
	xd := x.Data()
	xi := x.Ind()

	k = xi[0]
	m = xd[k]

	for _, l := range xi[1:] {
		if v := xd[l]; v < m {
			m = v
			k = l
		}
	}

	return
}

// Max returns the value and location of the largest element in the elements referenced by iterator, x.
func Max(x nd.Iterator) (m float64, k int) {
	xd := x.Data()
	xi := x.Ind()

	k = xi[0]
	m = xd[k]

	for _, l := range xi[1:] {
		if v := xd[l]; v > m {
			m = v
			k = l
		}
	}

	return
}

/*
func Min(x nd.Array) (k int) {
	it := x.NewIter()
	if it == nil {
		it = nd.NewIter(x)
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
	it := x.NewIter()
	if it == nil {
		it = nd.NewIter(x)
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
*/
