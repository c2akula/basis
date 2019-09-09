package nd

import "fmt"

// Broadcast broadcasts the shapes of the Ndarrays x and y.
func Broadcast(x, y *Ndarray) error {
	if !isBroadcastable(x, y) {
		return fmt.Errorf("%s: %s", ErrBroadcast, ErrShape)
	}

	// find the smaller array
	var (
		bshp, bstr Shape
		// nmin       int
		ndb int
	)
	if x.size > y.size {
		// nmin = y.ndims
		ndb = x.ndims
	} else {
		ndb = y.ndims
	}

	bshp = make(Shape, ndb)
	broadcastshp(x.shape, x.size, y.shape, y.size, bshp)
	if y.size > x.size {
		bstr = broadcaststr(x.shape, x.strides, bshp)
		x.ndims = len(bshp)
		x.shape = bshp
		x.strides = bstr
		x.size = ComputeSize(x.shape)
	} else {
		bstr = broadcaststr(y.shape, y.strides, bshp)
		y.ndims = len(bshp)
		y.shape = bshp
		y.strides = bstr
		y.size = ComputeSize(y.shape)
	}
	return nil
}

func broadcastshp(x Shape, szx int, y Shape, szy int, b Shape) {
	ndx, ndy := len(x), len(y)
	ndb := 0
	if b == nil {
		if szx > szy {
			ndb = ndx
		} else {
			ndb = ndy
		}
		b = make(Shape, ndb)
	}

	// find the smaller shape
	if szx > szy {
		// copy y
		for k, j := ndy-1, ndb-1; k >= 0; k, j = k-1, j-1 {
			b[j] = y[k]
		}
		// compare and set dimensions
		for k, j := ndx-1, ndb-1; k >= 0; k, j = k-1, j-1 {
			if nb, nx := b[j], x[k]; nx > nb {
				b[j] = nx
			} else if nb == 0 {
				b[j] = 1
			}
		}

	} else {
		// copy x
		for k, j := ndx-1, ndb-1; k >= 0; k, j = k-1, j-1 {
			b[j] = x[k]
		}
		// compare and set dimensions
		for k, j := ndy-1, ndb-1; k >= 0; k, j = k-1, j-1 {
			if nb, ny := b[j], y[k]; ny > nb {
				b[j] = ny
			} else if nb == 0 {
				b[j] = 1
			}
		}
	}
}

func broadcaststr(shp, str, bshp Shape) (bstr Shape) {
	bstr = ComputeStrides(bshp)
	// Rules
	// 1. Put a 0 in bstr at the dimensions where shp == 1
	// 2. If len(bshp) > len(shp) => all dimensions outside
	// shp should be set to 0 in bshp.
	if nshp, nbshp := len(shp), len(bshp); nshp < nbshp {
		for k, j := nshp-1, nbshp-1; k >= 0; k, j = k-1, j-1 {
			bstr[j] = str[k]

			if shp[k] == 1 {
				bstr[j] = 0
			}
		}

		// set the strides at the dimensions outside shp to 0
		for i := 0; i < nbshp-nshp; i++ {
			bstr[i] = 0
		}
		return bstr
	}

	// 3. If len(bshp) == len(shp), bstr(shp == 1) = 0
	copy(bstr, str)
	for j, n := range shp {
		if n == 1 {
			bstr[j] = 0
		}
	}

	return
}

// isBroadcasted checks if a stride across any dimension is 0.
func (array *Ndarray) isBroadcasted() bool {
	for _, n := range array.strides {
		if n == 0 {
			return true
		}
	}
	return false
}

// check if dimensions are compatible
// 1. dimensions are equal, or
// 2. one of the dimensions is 1
func isBroadcastable(x, y *Ndarray) bool {
	ndx := x.ndims
	ndy := y.ndims
	n := ndx
	if ndx > ndy {
		n = ndy
	}
	for k, j := ndy-1, ndx-1; n != 0; k, j = k-1, j-1 {
		ny, nx := y.shape[k], x.shape[j]
		if ny != nx {
			if !(ny == 1 || nx == 1) {
				return false
			}
		}
		n--
	}
	return true
}
