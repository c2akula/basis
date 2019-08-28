package go_nd

import "github.com/c2akula/go.nd/nd"

// Copy copies min(dst.Len(), src.Len()) no. of elements from src into dst.
// Note: If dst and src are the same, then dst is returned unmodified.
func Copy(dst, src nd.Iterator) nd.Iterator {
	if dst == src {
		return dst
	}

	n := dst.Len()
	if src.Len() < dst.Len() {
		n = src.Len()
	}

	d := dst.Data()
	s := src.Data()

	di := dst.Ind()
	si := src.Ind()

	for i, k := range di[:n] {
		d[k] = s[si[i]]
	}

	return dst
}
