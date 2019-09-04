package basis

import (
	"math"

	"github.com/c2akula/basis/nd"
)

const (
	InvSqrt2Pi = 1.0 / (math.SqrtPi * math.Sqrt2)
)

// Llh computes the Log-Likelihood of a Normal Distribution with mean, mu and variance, sigma.
func Llh(x nd.Iterator, mu, sigma float64) (l float64) {
	is := -0.5 * (1.0 / (sigma * sigma))
	f := InvSqrt2Pi * (1.0 / math.Sqrt(sigma))

	xd := x.Data()
	xi := x.Ind()

	for _, k := range xi {
		sm := xd[k] - mu
		l += math.Log(f * math.Exp(sm*sm*is))
	}

	return
}

/*

import (
	"math"
	"math/rand"

	"github.com/c2akula/basis/nd"
)



// Apply applies a unary function fn to the data referenced by the iterator, it.
func Apply(it nd.Iterator, fn func(float64) float64) nd.Iterator {
	for ; !it.Done(); it.Next() {
		v := it.At()
		*v = fn(*v)
	}
	it.Reset()
	return it
}

// Shift shifts the data referenced by the iterator by the scalar, s.
// If s > 0, then data += s is performed.
// If s < 0, then data -= s is performed.
func Shift(it nd.Iterator, s float64) nd.Iterator {
	for ; !it.Done(); it.Next() {
		*it.At() += s
	}
	it.Reset()
	return it
}

func Exp(it nd.Iterator) nd.Iterator {
	for !it.Done() {
		v := it.At()
		*v = math.Exp(*v)
		it.Next()
	}
	it.Reset()
	return it
}

// Sq performs x.^2
func Sq(x nd.Iterator) nd.Iterator {
	var v *float64
	for !x.Done() {
		v = x.At()
		*v *= *v
		x.Next()
	}
	x.Reset()
	return x
}

// Shuffle pseudo-randomly reorders the elements in the array, in-place.
func Shuffle(x nd.Array) nd.Array {
	it := x.NewIter()
	if it == nil {
		it = nd.NewIter(x)
	}
	xd := x.Data()
	rand.Shuffle(x.Size(), func(i, j int) {
		k, l := it.Seek(i), it.Seek(j)
		xd[k], xd[l] = xd[l], xd[k]
	})
	return x
}

// All returns the locations of the elements
// which satisfy the fn provided.
func All(it nd.Iterator, fn func(float64) bool) nd.Index {
	ind := make(nd.Index, 0, it.Len())
	for ; !it.Done(); it.Next() {
		if v := *it.At(); fn(v) {
			ind = append(ind, it.I())
		}
	}
	it.Reset()
	return ind
}

// Any finds the first occurence of the element in the data referenced
// by the iterator, it. If no element is found, then a (0,-1) tuple is
// returned.
func Any(it nd.Iterator, fn func(float64) bool) (v float64, k int) {
	for ; !it.Done(); it.Next() {
		if v = *it.At(); fn(v) {
			k = it.I()
			it.Reset()
			return
		}
	}
	it.Reset()
	return 0.0, -1
}

// Nonzeros returns the locations of the nonzero elements.
func Nonzeros(it nd.Iterator) nd.Index {
	return All(it, func(f float64) bool {
		return f != 0
	})
}

// Transform applies the function, fn to the elements that satifies the given predicate, pred.
func Transform(it nd.Iterator, pred func(float64) bool, fn func(float64) float64) nd.Iterator {
	for ; !it.Done(); it.Next() {
		if v := it.At(); pred(*v) {
			*v = fn(*v)
		}
	}
	it.Reset()
	return it
}
*/
