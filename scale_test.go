package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestScale(t *testing.T) {
	// a := Reshape(Arange(0, 60), Shape{3,4,5})
	a := nd.Reshape(nd.Arange(0, float64(nd.ComputeSize(TestArrayShape))), TestArrayShape)
	ait, _, _ := iter.New(a)
	v := 2.0

	exp := nd.Zeroslike(a)
	eit, ed, ei := iter.New(exp)
	Copy(eit, ait)
	Apply(exp.Take(), func(f float64) float64 {
		return f * v
	})

	b := nd.Zeroslike(a)
	bit, bd, _ := iter.New(b)
	Scale(v, ait, bit) // b <- v*a

	for _, k := range ei {
		if ed[k] != bd[k] {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
	}
}

func TestScaleView(t *testing.T) {
	av := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5}).View(nd.Index{1, 0, 1}, nd.Shape{2, 2, 3})
	ait, _, _ := iter.New(av)
	v := 2.0

	exp := nd.New(nd.Shape{2, 2, 3}, []float64{
		42, 44, 46,
		52, 54, 56,
		82, 84, 86,
		92, 94, 96,
	})
	_, ed, ei := iter.New(exp)

	b := nd.Zeroslike(av)
	bit, bd, _ := iter.New(b)
	Scale(v, ait, bit)

	for _, k := range ei {
		if ed[k] != bd[k] {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
	}
}

func BenchmarkScale(b *testing.B) {
	x := nd.Rand(TestArrayShape)
	it, _, _ := iter.New(x)
	v := 1.1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(v, it, it)
	}
}
