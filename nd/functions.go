package nd

import (
	"math"
)

// Fill writes the value v into the data referenced by the iterator, it.
func Fill(it Iterator, v float64) Iterator {
	for !it.Done() {
		*it.Upk() = v
		it.Next()
	}
	it.Reset()
	return it
}

// Apply applies a unary function fn to the data referenced by the iterator, it.
func Apply(it Iterator, fn func(float64) float64) Iterator {
	for ; !it.Done(); it.Next() {
		v := it.Upk()
		*v = fn(*v)
	}
	it.Reset()
	return it
}

// Sum returns the sum of the data referenced by the iterator, it.
func Sum(it Iterator) (s float64) {
	for ; !it.Done(); it.Next() {
		s += *it.Upk()
	}
	it.Reset()
	return
}

// Copy copies src into dst.
// Note: If dst or src is a view iterator
// then, the effect of the copy will be visible in
// the array referenced by the iterators.
// Note: dst and src must be of the same length.
func Copy(dst, src Iterator) Iterator {
	if dst.Len() != src.Len() {
		panic("dst and src must be of the same size")
	}

	for ; !dst.Done(); dst.Next() {
		*dst.Upk() = *src.Upk()
		src.Next()
	}
	dst.Reset()
	src.Reset()
	return dst
}

// Scale scales the data referenced by the iterator with the scalar, s.
func Scale(it Iterator, s float64) Iterator {
	for ; !it.Done(); it.Next() {
		*it.Upk() *= s
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
func Mean(it Iterator) (m float64) {
	return Sum(it) / float64(it.Len())
}

// Std computes the standard deviation of the data referenced by the iterator, it.
// It is equivalent to Sqrt(Var(it)).
func Std(it Iterator) (s float64) { return math.Sqrt(Var(it)) }

// Var computes the variance of the data referenced by the iterator, it.
func Var(it Iterator) (v float64) {
	mu := Mean(it)
	for !it.Done() {
		x := *it.Upk()
		v += (x - mu) * (x - mu)
		it.Next()
	}
	it.Reset()
	return v / float64(it.Len()-1)
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

func Dot(x, y Iterator) (f float64) {
	if x.Len() != y.Len() {
		panic("lengths of iterators must be equal")
	}
	for !y.Done() {
		f += *x.Upk() * *y.Upk()
		x.Next()
		y.Next()
	}
	x.Reset()
	y.Reset()
	return
}

func Axpy(a float64, x, y Iterator) Iterator {
	if x.Len() != y.Len() {
		panic("lengths of iterators must be equal")
	}

	switch a {
	case 0:
		return y
	case 1:
		for ; !y.Done(); y.Next() {
			*y.Upk() += *x.Upk()
			x.Next()
		}
	case -1:
		for ; !y.Done(); y.Next() {
			*y.Upk() -= *x.Upk()
			x.Next()
		}
	default:
		for ; !y.Done(); y.Next() {
			*y.Upk() += a * *x.Upk()
			x.Next()
		}
	}
	x.Reset()
	y.Reset()
	return y
}

// Axmy performs y *= a*x
func Axmy(a float64, x, y Iterator) Iterator {
	if x.Len() != y.Len() {
		panic("lengths of iterators must be equal")
	}

	switch a {
	case 0:
		return Fill(y, 0)
	case 1:
		for !y.Done() {
			*y.Upk() *= *x.Upk()
			x.Next()
			y.Next()
		}
	case -1:
		for !y.Done() {
			*y.Upk() *= -*x.Upk()
			x.Next()
			y.Next()
		}
	default:
		for !y.Done() {
			*y.Upk() *= a * *x.Upk()
			x.Next()
			y.Next()
		}
	}

	x.Reset()
	y.Reset()
	return y
}

// Axdy performs y /= a*x
func Axdy(a float64, x, y Iterator) Iterator {
	if x.Len() != y.Len() {
		panic("lengths of iterators must be equal")
	}

	switch a {
	case 0:
		return Fill(y, math.NaN())
	case 1:
		for !y.Done() {
			*y.Upk() /= *x.Upk()
			x.Next()
			y.Next()
		}
	case -1:
		for !y.Done() {
			*y.Upk() /= -*x.Upk()
			x.Next()
			y.Next()
		}
	default:
		for !y.Done() {
			*y.Upk() /= a * *x.Upk()
			x.Next()
			y.Next()
		}
	}

	x.Reset()
	y.Reset()
	return y
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
func Add(y Iterator, x ...Iterator) Iterator {
	if len(x) < 1 {
		return y
	}

	if len(x) == 1 {
		return Axpy(1, x[0], y)
	}

	if len(x) == 2 {
		return Axpy(1, x[1], Axpy(1, x[0], y))
	}

	for ; !y.Done(); y.Next() {
		v := y.Upk()
		for _, it := range x {
			*v += *it.Upk()
		}

		// increment each iterator
		for _, it := range x {
			it.Next()
		}
	}

	for _, it := range x {
		it.Reset()
	}
	y.Reset()
	return y
}

// Sub performs y -= x[0] -...- x[1] - x[0]
func Sub(y Iterator, x ...Iterator) Iterator {
	if len(x) < 1 {
		return y
	}

	if len(x) == 1 {
		return Axpy(-1, x[0], y)
	}

	if len(x) == 2 {
		return Axpy(-1, x[1], Axpy(-1, x[0], y))
	}

	for ; !y.Done(); y.Next() {
		v := y.Upk()
		for _, it := range x {
			*v -= *it.Upk()
		}

		// increment each iterator
		for _, it := range x {
			it.Next()
		}
	}

	for _, it := range x {
		it.Reset()
	}
	y.Reset()
	return y
}

// Mul performs y *= x[0] *...* x[1] * x[0]
func Mul(y Iterator, x ...Iterator) Iterator {
	if len(x) < 1 {
		return y
	}

	if len(x) == 1 {
		return Axmy(1, x[0], y)
	}

	if len(x) == 2 {
		return Axmy(1, x[1], Axpy(1, x[0], y))
	}

	for ; !y.Done(); y.Next() {
		v := y.Upk()
		for _, it := range x {
			*v *= *it.Upk()
		}

		// increment each iterator
		for _, it := range x {
			it.Next()
		}
	}

	for _, it := range x {
		it.Reset()
	}
	y.Reset()
	return y
}
