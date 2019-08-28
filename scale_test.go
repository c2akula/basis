package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestScale(t *testing.T) {
	// a := Reshape(Arange(0, 60), Shape{3,4,5})
	a := nd.Reshape(nd.Arange(0, float64(nd.ComputeSize(TestArrayShape))), TestArrayShape)
	ait := a.Range()
	v := 2.0

	exp := nd.Zeroslike(a)
	eit := exp.Range()
	ed, ei := eit.Iter()
	Copy(eit, ait)
	for _, k := range ei {
		ed[k] *= v
	}

	b := nd.Zeroslike(a)
	bit, bd := b.Range(), b.Data()
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
	ait := av.Range()
	v := 2.0

	exp := nd.New(nd.Shape{2, 2, 3}, []float64{
		42, 44, 46,
		52, 54, 56,
		82, 84, 86,
		92, 94, 96,
	})
	eit, ed := exp.Range(), exp.Data()
	ei := eit.Ind()

	b := nd.Zeroslike(av)
	bit, bd := b.Range(), b.Data()
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
	it := x.Range()
	v := 1.1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(v, it, it)
	}
}
