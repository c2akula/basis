package nd

import (
	"testing"
)

var RandArray = Rand(TestArrayShape)

func BenchmarkIterConcreteRange(b *testing.B) {
	b.ReportAllocs()
	a := RandArray
	ad, ai := a.Range().Iter()
	s := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, k := range ai {
			ad[k] *= 5.5
		}
	}
	_ = s * s
}

func TestIter_Iter(t *testing.T) {
	a := Rand(Shape{3, 4, 5})
	ait := a.Iter()
	for i, k := range ait.Ind() {
		if i != k {
			t.Logf("test 'Iter_Iter' failed. exp: %d, got: %d\n", i, k)
			t.Fail()
		}
	}

	av := a.View(Index{1, 0, 1}, Shape{1, 2, 3})
	ait = av.Iter()
	avstr := ComputeStrides(av.Shape())
	exp := make(Index, av.Size())
	ind := make(Index, av.Ndims())
	for i := range exp {
		exp[i] = Sub2ind(av.Strides(), Ind2sub(avstr, i, ind))
	}
	for i, k := range ait.Ind() {
		if exp[i] != k {
			t.Logf("test 'Iter_Iter' failed. exp: %d, got: %d\n", exp, ait.Ind())
			t.Fail()
		}
	}
}
