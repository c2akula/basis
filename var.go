package basis

import (
	"math"

	"github.com/c2akula/basis/nd"
)

func Var(x nd.Iterator) (s float64) {
	mu := Mean(x)

	xd := x.Data()
	xi := x.Ind()

	for _, k := range xi {
		d := xd[k] - mu
		s += d * d
	}

	return s * (1.0 / float64(x.Len()-1))
}

// Std computes the standard deviation of the array.
func Std(x nd.Iterator) (s float64) { return math.Sqrt(Var(x)) }
