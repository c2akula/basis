package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestMax(t *testing.T) {
	a := nd.New(nd.Shape{5, 3}, []float64{
		8, 5, 10,
		2, 4, 13,
		14, 1, 9,
		12, 3, 7,
		6, 11, 0,
	})
	expMax, expK := 14.0, 6
	v, k := Max(a.Iter())
	if expMax != v || expK != k {
		t.Logf("test failed. exp=[max: %v, k: %v], got=[max: %v, k: %v]\n", expMax, expK, v, k)
		t.Fail()
	}
}

func TestMin(t *testing.T) {
	a := nd.New(nd.Shape{5, 3}, []float64{
		8, 5, 10,
		2, 4, 13,
		14, 1, 9,
		12, 3, 7,
		6, 11, 0,
	})
	expMin, expK := 0.0, 14
	v, k := Min(a.Iter())
	if expMin != v || expK != k {
		t.Logf("test failed. exp=[min: %v, k: %v], got=[min: %v, k: %v]\n", expMin, expK, v, k)
		t.Fail()
	}
}
