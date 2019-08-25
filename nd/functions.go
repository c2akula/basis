package nd

import (
	"math"
	"math/rand"
)

// Apply applies a unary function fn to the data referenced by the iterator, it.
func Apply(it Iterator, fn func(float64) float64) Iterator {
	for ; !it.Done(); it.Next() {
		v := it.Upk()
		*v = fn(*v)
	}
	it.Reset()
	return it
}

// Shift shifts the data referenced by the iterator by the scalar, s.
// If s > 0, then data += s is performed.
// If s < 0, then data -= s is performed.
func Shift(it Iterator, s float64) Iterator {
	for ; !it.Done(); it.Next() {
		*it.Upk() += s
	}
	it.Reset()
	return it
}

// Mean computes the average of the elements pointed to
// by the iterator it.
func Mean(x Array) (m float64) {
	return Sum(x) * (1.0 / float64(x.Size()))
}

const (
	InvSqrt2Pi = 1.0 / (math.SqrtPi * math.Sqrt2)
)

// Llh computes the Log Likelihood of a Normal Distribution with mean, mu and variance, sigma.
func Llh(it Iterator, mu, sigma float64) (l float64) {
	is := -0.5 / (sigma * sigma)
	f := InvSqrt2Pi / math.Sqrt(sigma)
	for !it.Done() {
		s := *it.Upk()
		sm := s - mu
		l += math.Log(f * math.Exp(sm*sm*is))
		it.Next()
	}
	it.Reset()
	return l
}

func Exp(it Iterator) Iterator {
	for !it.Done() {
		v := it.Upk()
		*v = math.Exp(*v)
		it.Next()
	}
	it.Reset()
	return it
}

// Sq performs x.^2
func Sq(x Iterator) Iterator {
	var v *float64
	for !x.Done() {
		v = x.Upk()
		*v *= *v
		x.Next()
	}
	x.Reset()
	return x
}

// Add performs y += x[0] +...+ x[1] + x[0]
func Add(y Array, x ...Array) Array {
	if len(x) < 1 {
		return y
	}

	if len(x) == 1 {
		return Axpy(1, x[0], y)
	}

	if len(x) == 2 {
		return Axpy(1, x[1], Axpy(1, x[0], y))
	}

	return y
}

// Sub performs y -= x[0] -...- x[1] - x[0]
func Sub(y Array, x ...Array) Array {
	if len(x) < 1 {
		return y
	}

	if len(x) == 1 {
		return Axpy(-1, x[0], y)
	}

	if len(x) == 2 {
		return Axpy(-1, x[1], Axpy(-1, x[0], y))
	}

	return y
}

// Norm computes the Euclidean norm of the array, x.
func Norm(x Array) (n float64) { return math.Sqrt(Dot(x, x)) }

// Max returns the value and location of the largest element
// in the data referenced by the iterator.
func Max(it Iterator) (v float64, k int) {
	v = *it.Upk()
	it.Next()
	for ; !it.Done(); it.Next() {
		if max := *it.Upk(); max > v {
			v = max
			k = it.K()
		}
	}
	it.Reset()
	return
}

// Min returns the value and location of the smallest element
// in the data referenced by the iterator.
func Min(it Iterator) (v float64, k int) {
	v = *it.Upk()
	it.Next()
	for ; !it.Done(); it.Next() {
		if min := *it.Upk(); min < v {
			v = min
			k = it.K()
		}
	}
	it.Reset()
	return
}

// Shuffle pseudo-randomly reorders the elements in the array.
func Shuffle(it Iterator) Iterator {
	array := it.(*iterator).array
	rand.Shuffle(array.Size(), func(i, j int) {
		k, l := it.At(i), it.At(j)
		v1, v2 := array.Get(k), array.Get(l)
		v1, v2 = v2, v1
		array.Set(v1, k)
		array.Set(v2, l)
	})
	return it
}

// All returns the locations of the elements
// which satisfy the fn provided.
func All(it Iterator, fn func(float64) bool) Index {
	ind := make(Index, 0, it.Len())
	for ; !it.Done(); it.Next() {
		if v := *it.Upk(); fn(v) {
			ind = append(ind, it.K())
		}
	}
	it.Reset()
	return ind
}

// Any finds the first occurence of the element in the data referenced
// by the iterator, it. If no element is found, then a (0,-1) tuple is
// returned.
func Any(it Iterator, fn func(float64) bool) (v float64, k int) {
	for ; !it.Done(); it.Next() {
		if v = *it.Upk(); fn(v) {
			k = it.K()
			it.Reset()
			return
		}
	}
	it.Reset()
	return 0.0, -1
}

// Nonzeros returns the locations of the nonzero elements.
func Nonzeros(it Iterator) Index {
	return All(it, func(f float64) bool {
		return f != 0
	})
}

// Transform applies the function, fn to the elements that satifies the given predicate, pred.
func Transform(it Iterator, pred func(float64) bool, fn func(float64) float64) Iterator {
	for ; !it.Done(); it.Next() {
		if v := it.Upk(); pred(*v) {
			*v = fn(*v)
		}
	}
	it.Reset()
	return it
}
